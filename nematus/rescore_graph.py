'''
Rescoring a Moses word graph using a translation model.

The Moses word graph is obtained by adding the flags "-osg FILE" to Moses. Its format is

1 hyp=0 stack=0
1 hyp=119 stack=1 back=0 score=-1.158 transition=-1.158 forward=141 fscore=-1.788 covered=2-2 out=all @-@
1 hyp=117 stack=1 back=0 score=-1.299 transition=-1.299 forward=201 fscore=-1.758 covered=2-2 out=everything the
1 hyp=977 stack=2 back=76 score=-1.721 transition=-1.221 recombined=1457 forward=4021 fscore=-1.405 covered=2-2 out=everything the
...
'''
import os
import re
import sys
import argparse
import tempfile

import numpy
import json

from data_iterator import TextIterator
from util import load_dict, load_config
from alignment_util import *
from compat import fill_options
from collections import defaultdict

from theano_util import (load_params, init_theano_params)
from nmt import (pred_probs, build_model, prepare_data, init_params)

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import theano

from arcscorer import ArcScorer
from lattice import *

def main(models, source_file, graph_file, saveto, b=80,
         normalize=False, verbose=False, alignweights=False):

    sourcelines = source_file.readlines()

    scorer = None
    if models is not None and len(models) > 0:
        scorer = ArcScorer(models[0])

    graph_files = []
    if '{}' in graph_file:
        sentno = 0
        while os.path.exists(graph_file.format(sentno)):
            graph_files.append(graph_file.format(sentno))
            sentno += 1
    else:
        graph_files.append(graph_file)

    for i,graph_file in enumerate(graph_files):
        graph = read_graph(open(graph_file), i)

        if (scorer):
            scorer.set_source_sentence(sourcelines[i])

        graph.walk(scorer)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', type=int, default=80,
                        help="Minibatch size (default: %(default)s))")
    parser.add_argument('-n', action="store_true",
                        help="Normalize scores by sentence length")
    parser.add_argument('-v', action="store_true", help="verbose mode.")
    parser.add_argument('--models', '-m', type=str, nargs = '+', required=False,
                        help="model to use. Provide multiple models (with same vocabulary) for ensemble decoding")
    parser.add_argument('--source', '-s', type=argparse.FileType('r'),
                        required=True, metavar='PATH',
                        help="Source text file")
    parser.add_argument('--input', '-i', type=str,
                        default='-', help="Input graph or graph pattern (default: standard input)")
    parser.add_argument('--output', '-o', type=str,
                        default='-', help="Output file (default: standard output)")
    parser.add_argument('--walign', '-w',required = False,action="store_true",
                        help="Whether to store the alignment weights or not. If specified, weights will be saved in <input>.alignment")

    args = parser.parse_args()

    main(args.models, args.source, args.input,
         args.output, b=args.b, normalize=args.n, verbose=args.v, alignweights=args.walign)
