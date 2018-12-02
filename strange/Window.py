#!/usr/bin/env python

from __future__ import print_function

import os
import numpy as np
import subprocess


# # suppress the terrible h5 warning
# import warnings
# with warnings.catch_warnings(): 
#     warnings.filterwarnings("ignore", category=FutureWarning)
#     import h5py


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


