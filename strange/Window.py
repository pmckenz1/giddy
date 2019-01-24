#!/usr/bin/env python

from __future__ import print_function

import os
import sys
import time
import datetime
import subprocess
import tempfile
import numpy as np
import pandas as pd
import toytree
from numba import jit
from collections import Counter
from copy import deepcopy

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
	def __init__(self, name, workdir, ipyclient=None):
	    # setup and check path to data files
	    self.name = name
	    self.workdir = os.path.realpath(os.path.expanduser(workdir))
	    assert os.path.exists(os.path.join(self.workdir, name + ".hdf5")), (
	        "Results files not found in: {}/{}.*".
	        format(self.workdir, self.name))

	    # connect to parallel client
	    self.ipyclient = ipyclient

	    # find raxml binary relative to this file
	    strange_path = os.path.dirname(os.path.dirname(__file__))
	    bins_path = os.path.join(strange_path, "bins")
	    platform = ("linux" if "linux" in sys.platform else "macos")
	    self.raxml_binary = os.path.realpath(os.path.join(
	        bins_path,
	        'raxml-ng_v0.7.0_{}_x86_64'.format(platform)
	    ))
	    self.mb_binary = os.path.realpath(os.path.join(
	        bins_path,
	        'mb'
	    ))
	    self.coal_binary = os.path.realpath(os.path.join(
	        bins_path,
	        'hybrid-coal-{}'.format(platform)
	    ))

	    # load Coalseq results from paths
	    self.tree = toytree.tree(os.path.join(self.workdir, name + ".tree_ids.newick"))
	    self.database = os.path.join(self.workdir, name + ".hdf5")
	    self.clade_table = pd.read_csv(
	        os.path.join(self.workdir, name + ".clade_table.csv"), index_col=0)
	    self.tree_table = pd.read_csv(
	        os.path.join(self.workdir, name + ".tree_table.csv"), index_col=0)

	    # new attrs to fill
	    self.raxml_table = pd.DataFrame({})
	    self.mb_database = None
	    self.mb_database_mstrees = None
	    self.snames = None
	    self.seqarr = None
	    
	    # run functions
	    self.parse_seqnames()
	    
	    
	def parse_seqnames(self):
	    "read in seqarray as an ndarray"
	    with h5py.File(self.database, 'r') as io5:           
	        self.snames = io5.attrs["names"]


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
	    starts = range(0, dims[1] - window_size, slide_interval)
	    stops = range(window_size, dims[1], slide_interval)
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
	            args = (self.raxml_binary, self.database, start, stop)
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
	                break

	    # non-parallel code
	    else:
	        pass

	    # save raxml_table to disk
	    self.raxml_table.to_csv(
	        os.path.join(self.workdir, self.name + ".raxml_table"))


	def run_mb_sliding_windows(self, window_size, slide_interval,ipyclient=None):
	    """
	    Write temp nexus files, infer mb trees, store in hdf5, cleanup.
	    Pull chunks of sequence from seqarray in windows defined by tree_table
	    format as nexus and pass to mb subprocess.
	    """
	    # open h5 view to file
	    with h5py.File(self.database, 'r') as io5:
	        dims = io5["seqarr"].shape

	    # how many windows in chromosome? 
	    nwindows = int((dims[1]) / slide_interval)
	    assert nwindows, "No windows in data"

	    # get all intervals in a generator
	    starts = np.array(range(0, dims[1], slide_interval))
	    stops = starts + window_size
	    if stops[-1] > dims[1]:
	        stops[-1] = dims[1]
	    intervals = zip(starts, stops)

	    # start a hdf5 file for holding sliding window results
	    self.mb_database = self.workdir + '/' + self.name + '_mb.hdf5'
	    mbdb = h5py.File(self.mb_database)

	    mbdb.create_dataset('_intervals', data=np.array(intervals))

	    # start a tempdir
	    tempfile.gettempdir()

	    # parallelize tree inference
	    if ipyclient:
	        self.ipyclient = ipyclient
	    if self.ipyclient:
	        time0 = time.time()
	        lbview = self.ipyclient.load_balanced_view()

	        # infer trees by pulling in sequence from hdf5 on remote engines
	        rasyncs = {}
	        for idx, (start, stop) in enumerate(intervals):
	            args = (self.mb_binary, self.database, start, stop)
	            rasyncs[idx] = lbview.apply(run_mb, *args)

	        # track progress and collect results.
	        done = 0
	        while 1:
	            finished = [i for i in rasyncs if rasyncs[i].ready()]
	            for idx in finished:
	                if rasyncs[idx].successful():
	                    arr = rasyncs[idx].get()
	                    mbdb.create_dataset(str(idx), data=arr)
	                    del rasyncs[idx]
	                    done += 1
	                else:
	                    raise Exception(rasyncs[idx].get())
	            # progress
	            progressbar(done, nwindows, time0, "inferring mb trees")
	            time.sleep(0.5)
	            if not rasyncs:
	                break

	    # non-parallel code
	    else:
	        print("no engines started, shutting down...")
	        pass


	    nummbtrees = len(mbdb.keys())-1
	    fullset = set
	    for i in range(nummbtrees):
	        fullset = fullset.union(set(mbdb[str(i)][0]))
	    df = pd.DataFrame(list(enumerate(fullset)),columns=['idx','newick'])
	    df.to_csv(path_or_buf=self.workdir + '/' + self.name + '_mb_topokey.csv')

	    print('\nconsolidating...')

	    num_mb_gens = 4000

	    gtarr = np.zeros((num_mb_gens,nummbtrees),dtype=np.int32)

	    for column_idx in range(nummbtrees):
	        topos = np.array(list(mbdb[str(column_idx)]))[0]
	        topo_arr = np.zeros((topos.shape),dtype=np.int)
	        for topoidx in range(len(topos)):
	            topo_arr[topoidx] = np.argmax(df['newick'] == topos[topoidx])

	        probs = (np.array(list(mbdb[str(column_idx)]))[1]).astype(float)
	        probsum = np.sum(probs)
	        for probidx in range(len(probs)):
	            probs[probidx] = probs[probidx]/probsum
	        probs = (probs*num_mb_gens).astype(int)
	        probs[0:(num_mb_gens-np.sum(probs))] = probs[0:(num_mb_gens-np.sum(probs))] + 1

	        col = np.zeros((num_mb_gens),dtype=np.int64)
	        counter = 0
	        for topoidx in range(len(topo_arr)):
	            col[counter:(counter+probs[topoidx])] = np.repeat(topo_arr[topoidx],probs[topoidx])
	            counter += probs[topoidx]
	        
	        gtarr[:,column_idx] = col
	    pd.DataFrame(gtarr).to_csv(path_or_buf=self.workdir + '/' + self.name + '_mb_mcmc.csv')

	    #tot_list = []
	    #for _ in range(nummbtrees):
	    #    trees = mbdb[str(_)][0]
	    #    probs = mbdb[str(_)][1]
	    #    for loop in range(len(trees)):
	    #        tot_list.append([_,np.argmax(df['newick'] == trees[loop]),np.float(probs[loop])])
	    #totdf = pd.DataFrame(tot_list,columns=['idx','topo_idx','prob'])

	    #totdf.to_csv(path_or_buf=self.workdir + '/' + self.name + '_mb_post.csv')
	    print('done.')
	    mbdb.close()


	def run_mb_mstrees(self,ipyclient=None):
	    """
	    Write temp nexus files, infer mb trees, store in hdf5, cleanup.
	    Pull chunks of sequence from seqarray in windows defined by tree_table
	    format as nexus and pass to mb subprocess.
	    """
	    # open h5 view to file
	    with h5py.File(self.database, 'r') as io5:
	        dims = io5["seqarr"].shape

	    # how many windows in chromosome? 
	    nwindows = np.sum(self.tree_table.length != 0)
	    assert nwindows, "No windows in data"

	    starts = self.tree_table.start
	    stops = self.tree_table.end

	    filterarr = np.array(((starts-stops) != 0))

	    # get all intervals in a generator
	    intervals = zip(self.tree_table.start[filterarr], 
	        self.tree_table.end[filterarr])

	    # start a hdf5 file for holding sliding window results
	    self.mb_database_mstrees = self.workdir + '/' + self.name + '_mb_mstrees.hdf5'
	    mbdb = h5py.File(self.mb_database_mstrees)

	    mbdb.create_dataset('_intervals', data=np.array(intervals))

	    # start a tempdir
	    tempfile.gettempdir()

	    # parallelize tree inference
	    if ipyclient:
	        self.ipyclient = ipyclient
	    if self.ipyclient:
	        time0 = time.time()
	        lbview = self.ipyclient.load_balanced_view()

	        # infer trees by pulling in sequence from hdf5 on remote engines
	        rasyncs = {}
	        for idx, (start, stop) in enumerate(intervals):
	            args = (self.mb_binary, self.database, start, stop)
	            rasyncs[idx] = lbview.apply(run_mb, *args)

	        # track progress and collect results.
	        done = 0
	        while 1:
	            finished = [i for i in rasyncs if rasyncs[i].ready()]
	            for idx in finished:
	                if rasyncs[idx].successful():
	                    arr = rasyncs[idx].get()
	                    mbdb.create_dataset(str(idx), data=arr)
	                    del rasyncs[idx]
	                    done += 1
	                else:
	                    raise Exception(rasyncs[idx].get())
	            # progress
	            progressbar(done, nwindows, time0, "inferring mb trees on mstrees")
	            time.sleep(0.5)
	            if not rasyncs:
	                break

	    # non-parallel code
	    else:
	        print("no engines started, shutting down...")
	        pass


	    nummbtrees = len(mbdb.keys())-1
	    fullset = set
	    for i in range(nummbtrees):
	        fullset = fullset.union(set(mbdb[str(i)][0]))
	    df = pd.DataFrame(list(enumerate(fullset)),columns=['idx','newick'])
	    df.to_csv(
	            os.path.join(self.workdir, self.name + "_mb_mstrees_topokey.csv"))
	    print('\nconsolidating...')

	    num_mb_gens = 4000

	    gtarr = np.zeros((num_mb_gens,nummbtrees),dtype=np.int32)

	    for column_idx in range(nummbtrees):
	        topos = np.array(list(mbdb[str(column_idx)]))[0]
	        topo_arr = np.zeros((topos.shape),dtype=np.int)
	        for topoidx in range(len(topos)):
	            topo_arr[topoidx] = np.argmax(df['newick'] == topos[topoidx])

	        probs = (np.array(list(mbdb[str(column_idx)]))[1]).astype(float)
	        probsum = np.sum(probs)
	        for probidx in range(len(probs)):
	            probs[probidx] = probs[probidx]/probsum
	        probs = (probs*num_mb_gens).astype(int)
	        probs[0:(num_mb_gens-np.sum(probs))] = probs[0:(num_mb_gens-np.sum(probs))] + 1

	        col = np.zeros((num_mb_gens),dtype=np.int64)
	        counter = 0
	        for topoidx in range(len(topo_arr)):
	            col[counter:(counter+probs[topoidx])] = np.repeat(topo_arr[topoidx],probs[topoidx])
	            counter += probs[topoidx]
	        
	        gtarr[:,column_idx] = col
	    pd.DataFrame(gtarr).to_csv(path_or_buf=self.workdir + '/' + self.name + '_mb_mstrees_mcmc.csv')

	    #tot_list = []
	    #for _ in range(nummbtrees):
	    #    trees = mbdb[str(_)][0]
	    #    probs = mbdb[str(_)][1]
	    #    for loop in range(len(trees)):
	    #        tot_list.append([_,np.argmax(df['newick'] == trees[loop]),np.float(probs[loop])])
	    #totdf = pd.DataFrame(tot_list,columns=['idx','topo_idx','prob'])

	    #totdf.to_csv(path_or_buf=self.workdir + '/' + self.name + '_mb_mstrees_post.csv')
	    print('done.')
	    mbdb.close()


	def add_probs_topokey(self):
		topokey = pd.read_csv(
		            os.path.join(self.workdir, self.name + "_mb_mstrees_topokey.csv"),index_col=0)
		gtnewicks = topokey['newick']

		sptree=self.tree.newick
		problist = np.zeros(len(gtnewicks))
		time0 = time.time()
		ngts = len(gtnewicks)
		for i in range(ngts):
		    # run mb on nexus file
		    proc = subprocess.Popen([
		        self.coal_binary,
		        '-sp',
		        sptree,
		        '-gt',
		        gtnewicks[i]
		        ], 
		        stderr=subprocess.PIPE,
		        stdout=subprocess.PIPE,
		    )

		    statement = proc.stderr.read().strip()

		    # check for errors
		    out, _ = proc.communicate()
		    if proc.returncode:
		        raise Exception("hybrid-coal error: {}".format(out.decode()))
		    
		    problist[i] = np.float(statement.split(' = ')[1])
		    progressbar(i+1, ngts, time0, "computing gene tree probabilities")
		topokey.assign(probs=problist).to_csv(
		            os.path.join(self.workdir, self.name + "_mb_mstrees_topokey.csv"))


def run_mb(mb_binary, database, start, stop):
    "Build a temp nexus file, run mb and return array of newick trees and probs"
    
    # get sequence interval and count nsnps
    with h5py.File(database, 'r') as io5:
        names = io5.attrs["names"]
        seqs = io5["seqarr"][:, start:stop]

    # build nexus format string
    # make a string of sequences
    nexlist=""
    for i in range(len(names)):
        nexlist = nexlist + (names[i] + "".join(np.repeat(" ",13-len(names[i]))) + "".join(seqs[i])) + '\n'
    # write nexus to a tmp file
    fname = os.path.join(tempfile.gettempdir(), str(os.getpid()) + ".tmp")
    with open(fname, 'w') as temp:
        temp.write(

'''#NEXUS 

Begin data;
    Dimensions ntax={} nchar={};
    Format datatype=DNA gap=- missing=? matchchar=. interleave;
    Matrix

{}  ;
end;

begin mrbayes;
   set autoclose=yes nowarn=yes;
   lset nst=6 rates=invgamma;
   mcmc nruns=1 ngen=40000 samplefreq=100 diagnfreq=1000;
   sumt;
end;
'''.format(len(seqs),stop-start,nexlist)
        )

    # run mb on nexus file
    proc = subprocess.Popen([
        mb_binary,
        fname, 
        ], 
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )

    # check for errors
    out, _ = proc.communicate()
    if proc.returncode:
        raise Exception("mb error: {}".format(out.decode()))

    # read in inputs
    with open(fname+'.trprobs') as f:
        dat=f.read()
        dat = dat.split('\n')
        dat = [q.lstrip() for q in dat]
        dat = list(np.array(dat)[np.array([q[0:4] == 'tree' for q in dat])])
        trees = [q.split(' ')[-1] for q in dat]
        probs = [float(q.split('p = ')[1].split(', ')[0]) for q in dat]

    return np.array([trees,probs])


def run_raxml(raxml_binary, database, start, stop):
    "Build a temp phylip file, run raxml and return ML toytree"

    # get sequence interval and count nsnps
    with h5py.File(database, 'r') as io5:
        names = io5.attrs["names"]
        seqs = io5["seqarr"][:, start:stop]
        nsnps = np.invert(np.all(seqs == seqs[0], axis=0)).sum().astype(int)

    # build phylip format string
    phylip = ["{} {}".format(*seqs.shape).encode()]
    for name, seq in zip(names, seqs):
        phylip.append(name + b"\n".join(seq))

    # write phylip to a tmp file
    fname = os.path.join(tempfile.tempdir, str(os.getpid()) + ".tmp")
    with open(fname, 'w') as temp:
        temp.write(b"\n".join(phylip).decode())

    # run raxml on phylip file
    proc = subprocess.Popen([
        raxml_binary,
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
    return nsnps, toytree.tree(fname + ".raxml.bestTree").newick


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
            sd_normal):
    # record column, row numbers for arr
    numcols = arr.shape[1]
    numrows = arr.shape[0]
    
    # use normal distribution to sample which other (or same) column to draw from, for each column
    sc = np.array(list(range(numcols)) + np.random.normal(0,sd_normal,numcols).astype(int))
    # reflect around upper bound
    sc[sc >= numcols] = (numcols-1) + (numcols-1)-sc[sc >= numcols]
    sc[sc < 0] = np.abs(sc[sc < 0])
    
    # make new big array to sample from (columns correspond to columns to draw from)
    scarr = arr[sc]
    
    # make an array of the samples (ncols x mixnum) shape
    choicearr = np.zeros((numcols,mixnum),dtype=np.int64)
    for i in range(numcols):
        choicearr[i] = np.random.choice(scarr[:,i],size=mixnum)
        
    # make an array of idxs to replace (for each original column) with the new samples
    replace_idxs = np.random.randint(0,high=numrows,size=(choicearr.shape))
    
    # make the replacements
    for i in range(numcols):
        arr[:,i][replace_idxs[i]] = choicearr[i]

@jit
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

class MBmcmc:
	def __init__(self,
	    name,
	    workdir):
		self.name = name
		self.workdir = workdir

		self.mbcsv = np.array(pd.read_csv(os.path.join(self.workdir, 
			name + "_mb_mstrees_mcmc.csv")
			,index_col=0))
		self.topokey = pd.read_csv(os.path.join(self.workdir, 
			name + "_mb_mstrees_topokey.csv"),
			index_col=0)

		self.topoprobs = np.array(self.topokey['probs'])

	def update_x_times(self,
	                   mixnum,
	                   num_times,
	                   sd_normal):
	    for i in range(num_times):
	        replace(self.mbcsv,
	                mixnum,
	                sd_normal)

	def score(self):
		counted = Counter(mode(self.mbcsv)[0])
		expected = self.topoprobs[np.array(counted.keys())]
		observed = np.array(counted.values()).astype(float)/len(counted.values())
		return(rmse(expected,observed))

# from: https://stackoverflow.com/questions/16330831/most-efficient-way-to-find-mode-in-numpy-array
def mode(ndarray, axis=0):
    # Check inputs
    ndarray = np.asarray(ndarray)
    ndim = ndarray.ndim
    if ndarray.size == 1:
        return (ndarray[0], 1)
    elif ndarray.size == 0:
        raise Exception('Cannot compute mode on empty array')
    try:
        axis = range(ndarray.ndim)[axis]
    except:
        raise Exception('Axis "{}" incompatible with the {}-dimension array'.format(axis, ndim))

    # If array is 1-D and numpy version is > 1.9 numpy.unique will suffice
    if all([ndim == 1,
            int(np.__version__.split('.')[0]) >= 1,
            int(np.__version__.split('.')[1]) >= 9]):
        modals, counts = numpy.unique(ndarray, return_counts=True)
        index = np.argmax(counts)
        return modals[index], counts[index]

    # Sort array
    sort = np.sort(ndarray, axis=axis)
    # Create array to transpose along the axis and get padding shape
    transpose = np.roll(np.arange(ndim)[::-1], axis)
    shape = list(sort.shape)
    shape[axis] = 1
    # Create a boolean array along strides of unique values
    strides = np.concatenate([np.zeros(shape=shape, dtype='bool'),
                                 np.diff(sort, axis=axis) == 0,
                                 np.zeros(shape=shape, dtype='bool')],
                                axis=axis).transpose(transpose).ravel()
    # Count the stride lengths
    counts = np.cumsum(strides)
    counts[~strides] = np.concatenate([[0], np.diff(counts[~strides])])
    counts[strides] = 0
    # Get shape of padded counts and slice to return to the original shape
    shape = np.array(sort.shape)
    shape[axis] += 1
    shape = shape[transpose]
    slices = [slice(None)] * ndim
    slices[axis] = slice(1, None)
    # Reshape and compute final counts
    counts = counts.reshape(shape).transpose(transpose)[tuple(slices)] + 1

    # Find maximum counts and return modals/counts
    slices = [slice(None, i) for i in sort.shape]
    del slices[axis]
    index = np.ogrid[slices]
    index.insert(axis, np.argmax(counts, axis=axis))
    return sort[tuple(index)], counts[tuple(index)]


class MB_posts_csv:
    def __init__(self,
        posterior_path):
        self.mbcsv = pd.read_csv(posterior_path,index_col=0)

    def update_full(self,
                df,
                prop = .8,
                num_resamp = 300,
                scale = 1):
        tot_list = []
        num_focal = np.int(np.round(num_resamp*prop))
        num_non = num_resamp - num_focal
        nummbtrees = np.max(df['idx'])
    #    newfile=h5py.File('../tests/test.hdf5')
        for idx in range(nummbtrees):
            samp_idx = np.int(np.round(np.random.normal(idx,scale)))
            if samp_idx < 0 or samp_idx >= nummbtrees:
                samp_idx = idx + (samp_idx-idx)*-1

            newdraw = np.hstack([make_draw(df,idx,num_focal),make_draw(df, samp_idx, num_non)])
            counts = Counter(newdraw)
    #        newfile[str(idx)] = np.array(zip(np.repeat(0,len(counts)),
    #                                counts.keys(),
    #                                np.array(counts.values()).astype(float)/np.sum(counts.values())))
            tot_list.extend(zip(np.repeat(idx,len(counts)),
                                counts.keys(),
                                np.array(counts.values()).astype(float)/np.sum(counts.values())))
    #    newfile.close()
        return(pd.DataFrame(tot_list,columns=['idx','topo_idx','prob']))

class MB_posts_hdf5:
    # uses hdf5


    def __init__(self,
        posterior_path):
        mbinfs = h5py.File(name=posterior_path)
        self.posterior_list = [np.array(mbinfs[str(i)]) for i in range(len(mbinfs.keys())-1)]
        mbinfs.close()

    def resample_neighbors(self,
        num_times,
        prop,
        resamp_num=300):
        full_list = self.posterior_list
        for i in range(num_times):
            newfull_list = []
            for curridx in range(len(full_list)):
                ps = full_list[curridx][1]
                resamp = np.random.choice(len(ps),p=ps.astype(float)/np.sum(ps.astype(float)),size=np.int(resamp_num*prop))
                newsamped = full_list[curridx][0][resamp]
                if curridx != 0 and curridx != (len(full_list)-1):
                    leftright = np.array([curridx-1,curridx+1])
                elif curridx == 0:
                    leftright = np.array([curridx+1])
                else:
                    leftright = np.array([curridx-1])

                remaining = np.int(np.round((1-prop)*resamp_num))
                neighborsamps = []
                for neighbor in leftright:
                    ps = full_list[neighbor][1]
                    neighborsamps.extend(full_list[neighbor][0][np.random.choice(len(ps),p=ps.astype(float)/np.sum(ps.astype(float)),size=np.int(remaining/len(leftright)))])

                newsamped = np.hstack([newsamped,neighborsamps])
                countitem = Counter(newsamped)
                newfull_list.append([np.array(countitem.keys()),np.array(countitem.values()).astype(float)/np.sum(countitem.values())])
            del full_list
            full_list = deepcopy(newfull_list)
        return(full_list)

    def resample_index(self,full_list,orig_index,samp_index,prop,resamp_num=300):
        '''
        hand this a list (probably an adjusted copy of self.posterior list) to 
        resample a particular index
        '''
        newfull_list = []
        curridx = orig_index
        ps = full_list[curridx][1]
        resamp = np.random.choice(len(ps),p=ps.astype(float)/np.sum(ps.astype(float)),size=np.int(resamp_num*prop))
        newsamped = full_list[curridx][0][resamp]

        remaining = np.int(np.round((1-prop)*resamp_num))
        neighborsamps = []
        neighbor = samp_index
        ps = full_list[neighbor][1]
        neighborsamps.extend(full_list[neighbor][0][np.random.choice(len(ps),p=ps.astype(float)/np.sum(ps.astype(float)),size=np.int(remaining))])

        newsamped = np.hstack([newsamped,neighborsamps])
        countitem = Counter(newsamped)
        newfull_list.append([np.array(countitem.keys()),np.array(countitem.values()).astype(float)/np.sum(countitem.values())])
        return(newfull_list)


    def resample_normal(self,
                        num_times,
                        scale,
                        prop,
                        resamp_num):
        '''
        This resamples across all mb posterior distributions a specific number of times. 
        Sliding across all gene trees: For each tree, an integer is drawn from a normal distribution
        with mean of 0 and a specified variance. The corresponding posterior distribution away from the
        focal distribution is resampled from with a designated proportion.
        
        full_list: list of all posterior distributions of gene trees.
        num_times: number of times to resample all of the gene trees
        scale: the variance of the normal distribution used to draw a nearby tree to sample from
        prop: the proportion of resamples to be taken from the focal gene tree, as opposed to nearby trees
        resamp_num: the number of total resample draws, later normalized to floats between 0 and 1.
        '''
        full_list = self.posterior_list
        for _ in range(num_times):
            new_list = []
            listlen = len(full_list)
            for idx in range(listlen):
                newidx = self.get_samp_index(idx,scale)
                if newidx < 0 or newidx >= listlen:
                    newidx = idx + (newidx - idx)*(-1)
                new_list.append(
                    self.resample_index(full_list,
                           orig_index=idx,
                           samp_index=newidx,
                           prop = prop,
                           resamp_num=resamp_num)[0]
                )
            full_list = deepcopy(new_list)
        return(full_list)

    def get_samp_index(self,starting_idx, scale):
        return(np.int(starting_idx + np.round(np.random.normal(loc=0,scale=scale))))

@jit
def make_draw(self,df,idx,size):
    topos = df['topo_idx'][df['idx'] == idx]
    probs = df['prob'][df['idx'] == idx]
    probs = probs / np.sum(probs)
    return(np.random.choice(topos,p=probs,replace=True,size = size))


