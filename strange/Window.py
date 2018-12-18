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
        assert os.path.exists(os.path.join(self.workdir, name + ".newick")), (
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

        # load Coalseq results from paths
        self.tree = toytree.tree(os.path.join(self.workdir, name + ".newick"))
        self.database = os.path.join(self.workdir, name + ".hdf5")
        self.clade_table = pd.read_csv(
            os.path.join(self.workdir, name + ".clade_table.csv"), index_col=0)
        self.tree_table = pd.read_csv(
            os.path.join(self.workdir, name + ".tree_table.csv"), index_col=0)

        # new attrs to fill
        self.raxml_table = pd.DataFrame({})
        self.mb_database = None
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

        mbdb.close()


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


class Window:
    def __init__(self,
        full_seq_path):
        self.full_seq_path = full_seq_path


    def _write_subseq(self, output_seq_path, startidx, endidx):
        '''
        full_seq_path: string path to the hdf5 file containing the full alignment
        output_seq_path: path to the output file (ending in .fa). 
        startidx: integer index of site for window start
        endidx: integer index of site for window end
        
        Takes a full simulated alignment and writes out a user-defined window within that alignment.
        The outputs of this are .fa format and can be input to RAxML.
        '''
        seqs = h5py.File(self.full_seq_path)
        with open(output_seq_path,'w') as f:
            # make the header line telling how many taxa and how long the alignment is
            f.write(" "+str(seqs['alignment'].shape[0])+" "+str(endidx-startidx))
            f.write("\n")
            # for each row of the array, save a taxa ID and then the full sequence.
            for idx, seq in enumerate([''.join(i) for i in seqs['alignment'][:,startidx:endidx]]):
                # make a line to ID the taxon:
                f.write(str(idx+1) + ' '*(10-len(str(idx+1))))
                f.write("\n")
                #make a line for the sequence
                f.write(seq)
                f.write("\n")


    def produce_subseqs(self,
                       window_size,
                       slide_interval,
                       directory_name):
        '''
        user defines window size, sliding distance, path to the full alignment, and a
        directory in which to save the shortened sequences.
        '''
        # make a directory to save the sequence files
        if not os.path.exists(directory_name):
            os.mkdir(directory_name)
            print("Directory '" + directory_name +  "' created.")
        seqs = h5py.File(self.full_seq_path)
        total_len = seqs['alignment'].shape[1]
        seqs.close()
        # what's the farthest start index for our window
        startlimit = total_len-window_size

        index_nums = []
        index_starts = []
        index_ends = []
        
        startidx = 0
        num = 0
        while startidx <= startlimit:
            endidx = startidx + window_size
            
            # maintain an index for later reference
            index_nums.append(num)
            index_starts.append(startidx)
            index_ends.append(endidx)
            # write out a file for this loop's window
            self._write_subseq(directory_name+'/'+str(num)+'_'+str(startidx)+'_'+str(endidx)+'.fa',
                        startidx,
                        endidx)
            # then slide the window
            startidx += slide_interval
            num += 1
        
        # finish the process if there's a 'remainder'
        if startidx < total_len:
            
            # add to the index list
            index_nums.append(num)
            index_starts.append(startlimit)
            index_ends.append(total_len)
            self._write_subseq(directory_name+'/'+str(num)+'_'+str(startlimit)+'_'+str(total_len)+'.fa',
                    startlimit,
                    total_len)
        # write out the index file
        indexfile = h5py.File(directory_name+'/_index.hdf5')
        indexfile['nums'] = np.array(index_nums)
        indexfile['starts'] = np.array(index_starts)
        indexfile['ends'] = np.array(index_ends)


    def run_raxml(self, directory_name):
        '''
        directory_name: existing directory that is already created/filled using `produce_subseqs`
        runs raxml on each individual sequence in the directory produced with `produce_subseqs`
        uses magic right now but better to use subprocess
        '''
        names = os.listdir(directory_name)
        names = np.array(names)[np.array(names) != '_index.hdf5']
        for name in names:
            raxml = subprocess.Popen(['./raxml-ng','--msa', directory_name + '/' + name,'-model','GTR+G','-threads','2','--log','ERROR'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            raxml.communicate()


