#!/usr/bin/env python

import sys
import time
import datetime



class StrangeError(Exception):
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)


def progressbar(finished, total, start, message):
    progress = 100 * (finished / float(total))
    hashes = '#' * int(progress / 5.)
    nohash = ' ' * int(20 - len(hashes))
    elapsed = datetime.timedelta(seconds=int(time.time() - start))
    print("\r[{}] {:>3}% {} | {:<12} "
        .format(hashes + nohash, int(progress), elapsed, message),
        end="")
    sys.stdout.flush()    






# plotting tools to examine clade/node/tree support across regions...
# could look at the ipyrad.analysis.twisst code for this...
# ...
