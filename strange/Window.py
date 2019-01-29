#!/usr/bin/env python

from __future__ import print_function

import os
import re
import sys
import time
import pickle
import shutil
import tempfile
import subprocess

import numpy as np
import pandas as pd
import toytree

from .utils import progressbar

# suppress the terrible h5 warning
import warnings
with warnings.catch_warnings(): 
    warnings.filterwarnings("ignore", category=FutureWarning)
    import h5py


class SlidingWindow:
    """
    Perform tree inference on sliding windows over a phylip file and apply
    fuctions for comparing inferred trees to the true genealogies. Data 
    loaded in from the results of a Coalseq class object.   
    """
    def __init__(self, name, workdir="analysis-strange", ipyclient=None):
        # setup and check path to data files
        self.name = name
        self.workdir = os.path.realpath(os.path.expanduser(workdir))
        assert os.path.exists(os.path.join(self.workdir, name + ".hdf5")), (
            "Results files not found in: {}/{}.*".
            format(self.workdir, self.name))

        # connect to parallel client
        self.ipyclient = ipyclient

        # find raxml binary relative to this file
        self.get_bins()

        # load Coalseq results from paths
        self.tree = toytree.tree(os.path.join(self.workdir, name + ".newick"))
        self.database = os.path.join(self.workdir, name + ".hdf5")
        self.clade_table = pd.read_csv(
            os.path.join(self.workdir, name + ".clade_table.csv"), index_col=0)
        self.tree_table = pd.read_csv(
            os.path.join(self.workdir, name + ".tree_table.csv"), index_col=0)

        # new attrs to fill with raxml
        self.raxml_table = pd.DataFrame({})

        # new attrs to fill with mb   
        self.mb_table = pd.DataFrame({})
        self.mb_database = None
        self.mb_database_mstrees = None

        # sample names
        self.snames = None
        self.parse_seqnames()
        
        
    def parse_seqnames(self):
        "read in seqarray as an ndarray"
        with h5py.File(self.database, 'r') as io5:           
            self.snames = io5.attrs["names"]


    def get_bins(self):
        strange_path = os.path.dirname(os.path.dirname(__file__))
        bins_path = os.path.join(strange_path, "bins")
        platform = ("linux" if "linux" in sys.platform else "macos")
        self.raxml_binary = os.path.realpath(os.path.join(
            bins_path,
            'raxml-ng_v0.7.0_{}_x86_64'.format(platform)
        ))
        self.mb_binary = "mb"        
        self.coal_binary = os.path.realpath(os.path.join(
            bins_path,
            'hybrid-coal-{}'.format(platform)
        ))


    def run_raxml_sliding_windows(self, window_size, slide_interval):
        """
        Write temp phy files, infer raxml trees, store in table, cleanup.
        Pull chunks of sequence from seqarray in windows defined by tree_table
        format as phylip and pass to raxml subprocess.
        """
        # open h5 view to file
        with h5py.File(self.database, 'r') as io5:
            dims = io5["seqarr"].shape

        # how many windows in chromosome? 
        nwindows = int((dims[1] - window_size) / slide_interval)
        assert nwindows, "No windows in data"

        # get all intervals in a generator
        starts = range(0, dims[1], slide_interval)
        stops = range(window_size, dims[1] + window_size, slide_interval)
        intervals = zip(starts, stops)

        # setup table for results
        self.raxml_table["start"] = starts
        self.raxml_table["stop"] = stops
        self.raxml_table["nsnps"] = 0
        self.raxml_table["tree"] = None

        # parallelize tree inference
        if self.ipyclient:
            time0 = time.time()
            lbview = self.ipyclient.load_balanced_view()

            # infer trees by pulling in sequence from hdf5 on remote engines
            rasyncs = {}
            for idx, (start, stop) in enumerate(intervals):
                args = (self.raxml_binary, self.database, start, stop, idx)
                rasyncs[idx] = lbview.apply(run_raxml, *args)

            # track progress and collect results.
            done = 0
            while 1:
                finished = [i for i in rasyncs if rasyncs[i].ready()]
                for idx in finished:
                    if rasyncs[idx].successful():
                        nsnps, tree = rasyncs[idx].get()
                        self.raxml_table.loc[idx, "nsnps"] = nsnps
                        self.raxml_table.loc[idx, "tree"] = tree
                        del rasyncs[idx]
                        done += 1
                    else:
                        raise Exception(rasyncs[idx].get())
                # progress
                progressbar(done, nwindows, time0, "inferring raxml trees")
                time.sleep(0.5)
                if not rasyncs:
                    print("")
                    break

        # non-parallel code
        else:
            pass

        # save raxml_table to disk
        self.raxml_table.to_csv(
            os.path.join(self.workdir, self.name + ".raxml.csv"))


    def run_mb_sliding_windows(self, window_size, slide_interval):
        """
        Run sliding window tree inference in mrbayues. 
        Writes a pickled dictionary with all trees {0: tree1, 1: tree2 ...}
        Writes a numpy array with posterior tree distribution. 
        Writes a DataFrame with (nsnps, consenstre)
        """

        # ensure tmpdir for storing newicks
        mbdir = os.path.join(self.workdir, "mb-analysis-tmps")
        if not os.path.exists(mbdir):
            os.makedirs(mbdir)
        else:
            shutil.rmtree(mbdir)
            os.makedirs(mbdir)

        # open h5 view to file
        with h5py.File(self.database, 'r') as io5:
            dims = io5["seqarr"].shape

        # how many windows in chromosome? 
        nwindows = int((dims[1]) / slide_interval)
        assert nwindows, "No windows in data"

        # get all intervals in a generator
        # starts = range(0, dims[1] - window_size, slide_interval)
        starts = range(0, dims[1], slide_interval)        
        stops = range(window_size, dims[1] + window_size, slide_interval)
        intervals = zip(starts, stops)       

        # setup table for results
        self.mb_table["start"] = starts
        self.mb_table["stop"] = stops
        self.mb_table["nsnps"] = 0
        self.mb_table["tree"] = None

        # store resulting trprobs files
        treefiles = {}

        # parallelize tree inference
        if self.ipyclient:
            time0 = time.time()
            lbview = self.ipyclient.load_balanced_view()

            # infer trees by pulling in sequence from hdf5 on remote engines
            rasyncs = {}
            for idx, (start, stop) in enumerate(intervals):
                args = (self.mb_binary, self.database, start, stop, idx)
                rasyncs[idx] = lbview.apply(run_mb, *args)

            # track progress and collect results.
            done = 0
            while 1:
                finished = [i for i in rasyncs if rasyncs[i].ready()]
                for idx in finished:
                    if rasyncs[idx].successful():
                        nsnps, tree, treefile = rasyncs[idx].get()
                        self.mb_table.loc[idx, 'nsnps'] = nsnps
                        self.mb_table.loc[idx, 'tree'] = tree
                        treefiles[idx] = treefile
                        del rasyncs[idx]
                        done += 1
                    else:
                        raise Exception(rasyncs[idx].get())

                progressbar(done, nwindows, time0, "inferring mb trees")
                time.sleep(0.5)
                if not rasyncs:
                    print("")
                    break

        # non-parallel code
        else:
            pass

        # stream once through the treefiles to index every tree
        idx = 0
        treeset = {}
        for trfile in treefiles.values():
            with open(trfile) as indat:
                dat = indat.readlines()
                tlines = [i for i in dat if i.lstrip().startswith("tree ")]
                trees = [i.strip().split("] ")[-1] for i in tlines]
                for tree in trees:
                    if tree not in treeset.values():
                        treeset[idx] = tree
                        idx += 1

        # reverse lookup dictionary for trees
        revset = {j: i for (i, j) in treeset.items()}

        # stream second time through to fill array of observed trees 
        nsamples = 1000
        posterior = np.zeros((nsamples, nwindows), dtype=np.uint32)

        # Build database
        for tidx, trfile in treefiles.items():
            with open(trfile) as indat:
                dat = indat.readlines()
                tlines = [i for i in dat if i.lstrip().startswith("tree ")]
                trees = [i.strip().split("] ")[-1] for i in tlines]
                freqs = [i.split("=")[1][:-4] for i in tlines]
                counts = (np.array(freqs, dtype=float) * nsamples).astype(int)

                # fill column with tree indices. 
                idx = 0
                for tree, count in zip(trees[::-1], counts[::-1]):
                    treeidx = revset[tree]
                    posterior[idx:idx + count, tidx] = treeidx
                    idx += count

                # fill leftover with most common to accommodate rounding errors
                posterior[idx:, tidx] = treeidx

        # save tree mapping dictionary as a pickle
        phandle = os.path.join(self.workdir, self.name + ".mb.p")
        with open(phandle, 'wb') as pout:
            pickle.dump(treeset, pout)

        # save posterior array as a numpy pickle
        np.save(os.path.join(self.workdir, self.name + ".mb.npy"), posterior)

        # save tree_table as DataFrame
        self.mb_table.to_csv(os.path.join(self.workdir, self.name + ".mb.csv"))           

        # cleanup (comment to debug)
        shutil.rmtree(mbdir)


class TreeInference:
    def __init__(self, binary, database, start, stop, idx=None, **kwargs):
        "Class for running individual tree inferences on remote engines"

        # store args
        self.binary = binary
        self.database = database
        self.workdir = os.path.dirname(self.database)
        self.start = start
        self.stop = stop
        self.idx = idx
        self.kwargs = kwargs

        # parse names, seq region, and count nsnps
        with h5py.File(database, 'r') as io5:
            self.names = io5.attrs["names"]
            self.seqs = io5["seqarr"][:, start:stop]
            self.nsnps = np.invert(
                np.all(self.seqs == self.seqs[0], axis=0)).sum().astype(int)


    def run_raxml(self):
        "Build a temp phylip file, run raxml and return ML toytree"

        # build phylip format string
        phylip = ["{} {}".format(*self.seqs.shape).encode()]
        for name, seq in zip(self.names, self.seqs):
            phylip.append(name + b"\n".join(seq))

        # write phylip to a tmp file
        fname = os.path.join(tempfile.gettempdir(), str(os.getpid()) + ".tmp")
        with open(fname, 'w') as temp:
            temp.write(b"\n".join(phylip).decode())

        # run raxml on phylip file
        proc = subprocess.Popen([
            self.binary,
            '--msa', fname, 
            '--model', 'JC',  # 'GTR+G',
            '--threads', '1',
            '--redo',
            ], 
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )

        # check for errors
        out, _ = proc.communicate()
        if proc.returncode:
            raise Exception("raxml error: {}".format(out.decode()))
        return self.nsnps, toytree.tree(fname + ".raxml.bestTree").newick


    def run_mb(self):

        # mdict 
        mdict = {
            i.strip(): b"".join(j) for (i, j) in zip(self.names, self.seqs)
        }
        
        ## create matrix as a string
        max_name_len = max([len(i) for i in mdict])
        namestring = "{:<" + str(max_name_len + 1) + "} {}\n"
        matrix = ""
        for i in mdict.items():
            matrix += namestring.format(i[0].decode(), i[1].decode())

        # parameters
        ngen = (self.kwargs["ngen"] if self.kwargs.get("ngen") else 100000)
        sfreq = (self.kwargs["samplefreq"] if self.kwargs.get("samplefreq") else 100)
        burnin = (self.kwargs["burnin"] if self.kwargs.get("burnin") else 10000)

        # write nexus block
        mbdir = os.path.join(self.workdir, "mb-analysis-tmps")
        handle = os.path.join(mbdir, "{}.nex".format(self.idx))
        with open(handle, 'w') as outnex:
            outnex.write(
                NEXBLOCK.format(**{
                    "ntax": self.seqs.shape[0],
                    "nchar": self.seqs.shape[1], 
                    "matrix": matrix,
                    "ngen": ngen,
                    "sfreq": sfreq,
                    "burnin": burnin,
                    })
            )

        # run mrbayes       
        proc = subprocess.Popen(
            [self.binary, handle], 
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )

        # check for errors
        out, _ = proc.communicate()
        if proc.returncode:
            raise Exception("mb error: {}".format(out.decode()))

        # load the majority rule consensus tree 
        with open(handle + ".con.tre") as intre:
            dat = intre.readlines()
            treeline = [i for i in dat if i.lstrip().startswith("tree")][0]
            tree = treeline.split("[&U] ")[-1]
            tree = re.sub(r"\[(.*?)\]", "", tree)
            tree = toytree.tree(tree).write(fmt=0)

        # return nsnps
        return self.nsnps, tree, handle + ".trprobs"


# functions for remote calls of TreeInference class for ipyparallel
def run_raxml(binary, database, start, stop, idx):
    ti = TreeInference(binary, database, start, stop, idx)
    nsnps, tree = ti.run_raxml()
    return nsnps, tree


def run_mb(binary, database, start, stop, idx):
    ti = TreeInference(binary, database, start, stop, idx)
    nsnps = ti.run_mb()
    return nsnps

class MBmcmc:
    def __init__(self,
        name,
        workdir):
        self.name = name
        self.workdir = workdir

        self.mbcsv = np.array(pd.read_csv(os.path.join(self.workdir, 
            name + "_mb_mstrees_mcmc.csv")
            ,index_col=0),dtype=np.int32)
        self._mbcsv = deepcopy(self.mbcsv) # for updating
        self.topokey = pd.read_csv(os.path.join(self.workdir, 
            name + "_mb_mstrees_topokey.csv"),
            index_col=0)

        self.topoprobs = np.array(self.topokey['probs'])
        self.currscore = score(self.mbcsv,self.topoprobs)

        self.gtlist = []
        self.original_gts = mode(self.mbcsv)
        self.currgts = mode(self.mbcsv)

        self.zeros = np.zeros((self.mbcsv.shape),dtype = np.int32)

        # make the database name, and 
        self.db = os.path.join(self.workdir, 
            name + "_mcmc_res.hdf5")

    def update_x_times(self,
                       mixnum,
                       num_times,
                       sd_normal):
        for i in range(num_times):
            replace(self.mbcsv,
                    mixnum,
                    sd_normal,
                    self.zeros)

    def _update_and_score(self,
        mixnum,
        sd_normal,
        p,
        sample_freq,
        update_num,
        record_arr):
        replace(self._mbcsv,
                    mixnum,
                    sd_normal,
                    self.zeros)
        new_score, gts = score(self._mbcsv,self.topoprobs)
        if new_score < self.currscore:
            self.mbcsv = self._mbcsv
            self.currscore = new_score
            #self.gtlist.append(gts)
            self.currgts = gts
            if not (update_num+1)%sample_freq:
                record_arr[np.int16(np.floor(update_num/sample_freq))] = self.mbcsv
        elif np.random.binomial(1,p):
            self.mbcsv = self._mbcsv
            self.currscore = new_score
            #self.gtlist.append(gts)
            self.currgts = gts
            if not (update_num+1)%sample_freq:
                record_arr[np.int16(np.floor(update_num/sample_freq))] = self.mbcsv
        else:
            self._mbcsv = self.mbcsv
            #self.gtlist.append(self.currgts)
            if not (update_num+1)%sample_freq:
                record_arr[np.int16(np.floor(update_num/sample_freq))] = self.mbcsv

    def run_mcmc(self,
        numtimes=10000,
        sample_freq=100,
        batchsize=500,
        p=.1,
        mixnum=5,
        sd_normal=2):

        self.hdf5 = h5py.File(self.db,'a')
        if 'mcmcarr' not in self.hdf5.keys():
            self.hdf5.create_dataset('mcmcarr',
                shape=(0,self.mbcsv.shape[0],self.mbcsv.shape[1]),
                maxshape=(None,self.mbcsv.shape[0],self.mbcsv.shape[1]),
                dtype=np.int32)

        num_samps = np.int16(np.floor(numtimes/sample_freq))

        set_lengths = np.hstack([np.repeat(batchsize*sample_freq,np.int16(np.floor(numtimes/batchsize/sample_freq))),
            numtimes%(batchsize*sample_freq)])
        if set_lengths[-1] == 0:
            set_lengths = set_lengths[:-1]

        # progress bar
        time0 = time.time()
        counter = 1

        for set_size in set_lengths:
            setarr = np.zeros((np.int16(np.floor(set_size/sample_freq)), 
                self.mbcsv.shape[0], self.mbcsv.shape[1]))
            for update_num in np.arange(set_size):
                self._update_and_score(mixnum,
                    sd_normal,
                    p,
                    sample_freq,
                    update_num,
                    setarr)
                counter += 1
                progressbar(counter, numtimes, time0, "running mcmc")
            self.hdf5['mcmcarr'].resize((self.hdf5['mcmcarr'].shape[0] + setarr.shape[0]),axis = 0)
            self.hdf5['mcmcarr'][-setarr.shape[0]:] = setarr

        self.hdf5.close()

def progressbar(finished, total, start, message):
    progress = 100 * (finished / float(total))
    hashes = '#' * int(progress / 5.)
    nohash = ' ' * int(20 - len(hashes))
    elapsed = datetime.timedelta(seconds=int(time.time() - start))
    print("\r[{}] {:>3}% {} | {:<12} "
        .format(hashes + nohash, int(progress), elapsed, message),
        end="")
    sys.stdout.flush()    

@jit
def replace(arr,
            mixnum,
            sd_normal,
            sc):
    '''
    TO DO: make the the sc array outside the function, pass that in.

    sc argument is just an empty array.
    '''
    # record column, row numbers for arr
    numcols = arr.shape[1]
    numrows = arr.shape[0]

    # use normal distribution to sample which other (or same) column to draw from, for each column
    sc = np.array(np.arange(numcols) + np.random.normal(0,sd_normal,numcols).astype(np.int32))
    # reflect around upper bound
    sc[sc >= numcols] = (numcols-1) + (numcols-1)-sc[sc >= numcols]
    sc[sc < 0] = np.abs(sc[sc < 0])

    # make new big array to sample from (columns correspond to columns to draw from)
    scarr = arr[:,sc]

    # make an array of the samples (ncols x mixnum) shape
    choicearr = scarr[np.random.randint(0,10,(mixnum,scarr.shape[1])),np.arange(scarr.shape[1])].T

    # make an array of idxs to replace (for each original column) with the new samples
    replace_idxs = np.random.randint(0,high=numrows,size=(choicearr.shape))

    idxarr = np.zeros((numcols,mixnum),dtype=np.int16)
    for idx in np.arange(mixnum):
        idxarr[:,idx] = np.arange(numcols)

    arr[replace_idxs,idxarr] = choicearr

@jit
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def score(arr,topoprobs):
    treemode = mode(arr)
    counted = Counter(treemode)
    expected = topoprobs[np.array(counted.keys())]
    observed = np.array(counted.values()).astype(float)/len(counted.values())
    return([rmse(expected,observed),treemode])

@jit(nopython=True)
def mode(arr):
    outarr = np.zeros((arr.shape[1]),dtype = np.int32)
    for i in np.arange(arr.shape[1]):
        outarr[i] = np.argmax(np.bincount(arr[:,i]))
    return(outarr)


# GLOBALS ----------------------------------------------
NEXBLOCK = """\
#NEXUS
begin data;
dimensions ntax={ntax} nchar={nchar};
format datatype=dna interleave=yes gap=- missing=N;
matrix
{matrix}
    ;

begin mrbayes;
set autoclose=yes nowarn=yes;
lset nst=6 rates=gamma;
mcmc nruns=1 ngen={ngen} samplefreq={sfreq} printfreq={ngen};
sumt burnin={burnin};
end;
"""
