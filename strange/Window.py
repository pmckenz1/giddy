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
