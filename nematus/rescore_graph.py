'''
Rescoring a Moses word graph using a translation model.

The Moses word graph is obtained by adding the flags "-osg FILE" to Moses. Its format is

1 hyp=0 stack=0
1 hyp=119 stack=1 back=0 score=-1.158 transition=-1.158 forward=141 fscore=-1.788 covered=2-2 out=all @-@
1 hyp=117 stack=1 back=0 score=-1.299 transition=-1.299 forward=201 fscore=-1.758 covered=2-2 out=everything the
1 hyp=977 stack=2 back=76 score=-1.721 transition=-1.221 recombined=1457 forward=4021 fscore=-1.405 covered=2-2 out=everything the
...
'''
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

class Arc:
    head = None
    tail = None
    label = None
    score = None

    def __str__(self):
        return 'ARC[{} -> {} -> {}, {}]'.format(self.tail, self.label, self.head, self.score)

    def __init__(self, tail, label, head, score):
        self.tail = tail
        self.label = label
        self.head = head
        self.score = float(score)

class Node:
    def __init__(self, id = 0):
        self.id = id
        self.outarcs = []
        self.inarcs = []

    def __str__(self):
        return 'NODE[{}]'.format(self.id)

    def addOutgoingArc(self, arc):
        self.outarcs.append(arc)

    def addIncomingArc(self, arc):
        self.inarcs.append(arc)

    def getIncomingArcs(self):
        return self.inarcs

class Graph:
    sentno = -1
    root = None
    nodes = {}
    nodelist = []
    arccount = 0
    finalstate = -1

    def __str__(self):
        return `self.sentno`

    def __init__(self, sentno):
        self.sentno = sentno
        root = self.node(0)

    def id(self):
        return self.sentno

    def numnodes(self):
        return len(self.nodes)

    def numarcs(self):
        return self.arccount

    def addArc(self, tail, label, head, score):
        arc = Arc(self.node(tail), label, self.node(head), score)
        self.node(tail).addOutgoingArc(arc)
        self.node(head).addIncomingArc(arc)
        self.arccount += 1

    def node(self, id):
        id = int(id)
        if not self.nodes.has_key(id):
            node = Node(id)
            self.nodes[id] = node
            self.nodelist.append(node)

        return self.nodes[id]

    def score(self, state, arc):
        """The default scoring option. Returns the score read in on the arc, ignoring the old state and not returning a new one."""
        return None, arc.score

    def walk(self, scorer):

        # for each node, the best state, the word that produced it, and the cumulative score
        states = {}
        for node in self.nodelist:
#            print 'processing {} with {} incoming arcs'.format(node, len(node.getIncomingArcs()))
            best = (-999999, None, None)
            for arc in node.getIncomingArcs():
                oldscore, state, word = states.get(arc.tail, (0, None, ""))

                newstate, transitioncost = scorer.score(state, arc)
                score = oldscore + transitioncost

#                print '  {} -> {}'.format(arc, score)
                if score > best[0]:
#                    print '  new best ({} > {})'.format(score, best[0])
                    best = (score, newstate, arc)
                # else:
                #     print '  old is better ({} < {})'.format(score, best[0])

            if best[2] is not None:
                states[node] = best
                arc = best[2]
                # print 'best -> {} is {} ({})'.format(arc.head, arc.label, arc.score)

        node = self.node(self.finalstate)
        # print 'best modelscore =', states[node][0]
        seq = []
        while node.id != 0:
            score, state, arc = states.get(node)
            seq.insert(0, arc.label)
            node = arc.tail

        print states[self.node(self.finalstate)][0], ' '.join(seq).replace('<eps>','').replace('_', ' ').strip()

def walk_graph(graph, scorer):
    """
    score is a function parameterized as score(state, word) -> (nextstate, prob)

    the score() function should treat a null state as an initial state
    """
    graph.walk(graph)


def read_graph(search_graph_file, sentno = 0):
    """
    Reads an OpenFST file and constructs a walkable graph.
    """

    graph = Graph(sentno)
    for line in search_graph_file:
        try:
            tail, head, source, target = line.rstrip().split(' ', 3)
        except ValueError:
            graph.finalstate = int(line.rstrip())

        if ' ' in target:
            target, score = target.split(' ')
        else:
            score = 0.0

        tail = int(tail)
        head = int(head)
        graph.addArc(tail, target, head, -float(score))

#    print "graph[->{}] {} has {} nodes and {} arcs".format(graph.finalstate, graph.id(), graph.numnodes(), graph.numarcs())
    return graph


def rescore_model(source_file, search_graph_file, saveto, models, options, b, normalize, verbose, alignweights):

    trng = RandomStreams(1234)

    fs_log_probs = []

    for model, option in zip(models, options):

        # load model parameters and set theano shared variables
        params = numpy.load(model)
        tparams = init_theano_params(params)

        trng, use_noise, \
            x, x_mask, y, y_mask, \
            opt_ret, \
            cost = \
            build_model(tparams, option)
        inps = [x, x_mask, y, y_mask]
        use_noise.set_value(0.)

        if alignweights:
            sys.stderr.write("\t*** Save weight mode ON, alignment matrix will be saved.\n")
            outputs = [cost, opt_ret['dec_alphas']]
            f_log_probs = theano.function(inps, outputs)
        else:
            f_log_probs = theano.function(inps, cost)

        fs_log_probs.append(f_log_probs)

    def _score(pairs, alignweights=False):
        # sample given an input sequence and obtain scores
        scores = []
        alignments = []
        for i, f_log_probs in enumerate(fs_log_probs):
            score, alignment = pred_probs(f_log_probs, prepare_data, options[i], pairs, normalize=normalize, alignweights = alignweights)
            scores.append(score)
            alignments.append(alignment)

        return scores, alignments

    sourcelines = source_file.readlines()
    graph = read_graph(search_graph_file)
    walk_graph(graph)

def main(models, source_file, nbest_file, saveto, b=80,
         normalize=False, verbose=False, alignweights=False):

    # load model model_options
    options = []
    for model in models:
        options.append(load_config(model))

        fill_options(options[-1])

    graph = read_graph(nbest_file)
    walk_graph(graph, 0)

#    rescore_model(source_file, nbest_file, saveto, models, options, b, normalize, verbose, alignweights)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', type=int, default=80,
                        help="Minibatch size (default: %(default)s))")
    parser.add_argument('-n', action="store_true",
                        help="Normalize scores by sentence length")
    parser.add_argument('-v', action="store_true", help="verbose mode.")
    parser.add_argument('--models', '-m', type=str, nargs = '+', required=True,
                        help="model to use. Provide multiple models (with same vocabulary) for ensemble decoding")
    parser.add_argument('--source', '-s', type=argparse.FileType('r'),
                        required=True, metavar='PATH',
                        help="Source text file")
    parser.add_argument('--input', '-i', type=argparse.FileType('r'),
                        default=sys.stdin, metavar='PATH',
                        help="Input n-best list file (default: standard input)")
    parser.add_argument('--output', '-o', type=argparse.FileType('w'),
                        default=sys.stdout, metavar='PATH',
                        help="Output file (default: standard output)")
    parser.add_argument('--walign', '-w',required = False,action="store_true",
                        help="Whether to store the alignment weights or not. If specified, weights will be saved in <input>.alignment")

    args = parser.parse_args()

    main(args.models, args.source, args.input,
         args.output, b=args.b, normalize=args.n, verbose=args.v, alignweights=args.walign)
