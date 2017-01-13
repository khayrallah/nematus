#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import argparse
import cPickle as pkl
import json
import numpy
from theano_util import (load_params, init_theano_params)
from nmt import (build_sampler, pred_probs, build_model, prepare_data, init_params, gen_sample)
from util import load_dict
import theano

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
trng = RandomStreams(1234)
use_noise = theano.shared(numpy.float32(0.))

class ArcScorer(object):
    
    def __init__(self, model):
        '''Loads a Nematus NMT model
        Returns
        - f_init: NMT initialization function
        - f_next: NMT next function
        - word_dict: source mapping of word to id
        - word_dict_trag: target mapping of word to id
        - word_idict_trg: target mapping of id to word
        For all these, we assume there is only one version, i.e. no ensemble
        '''
        options = []
        try:
            print '%s.json' % model
            with open('%s.json' % model, 'rb') as f:
                options.append(json.load(f))
        except:
            with open('%s.pkl' % model, 'rb') as f:
                options.append(pkl.load(f))

        #hacks for using old models with missing options
        if not 'dropout_embedding' in options[-1]:
            options[-1]['dropout_embedding'] = 0
        if not 'dropout_hidden' in options[-1]:
            options[-1]['dropout_hidden'] = 0
        if not 'dropout_source' in options[-1]:
            options[-1]['dropout_source'] = 0
        if not 'dropout_target' in options[-1]:
            options[-1]['dropout_target'] = 0
        if not 'factors' in options[-1]:
            options[-1]['factors'] = 1
        if not 'dim_per_factor' in options[-1]:
            options[-1]['dim_per_factor'] = [options[-1]['dim_word']]

        dictionaries = options[0]['dictionaries']
        dictionaries_source = dictionaries[:-1]
        dictionary_target = dictionaries[-1]

        # load source dictionary and invert
        word_dicts = []
        word_idicts = []
        for dictionary in dictionaries_source:
            word_dict = load_dict(dictionary)
            if options[0]['n_words_src']:
                for key, idx in word_dict.items():
                    if idx >= options[0]['n_words_src']:
                        del word_dict[key]
            word_idict = dict()
            for kk, vv in word_dict.iteritems():
                word_idict[vv] = kk
            word_idict[0] = '<eos>'
            word_idict[1] = 'UNK'
            word_dicts.append(word_dict)
            word_idicts.append(word_idict)

        # load target dictionary and invert
        word_dict_trg = load_dict(dictionary_target)
        word_idict_trg = dict()
        for kk, vv in word_dict_trg.iteritems():
            word_idict_trg[vv] = kk
        word_idict_trg[0] = '<eos>'
        word_idict_trg[1] = 'UNK'

        option = options[0]
        params = init_params(option)
        params = load_params(model, params)
        tparams = init_theano_params(params)
        f_init, f_next = build_sampler(tparams, option, use_noise, trng, return_alignment=False)

        self.f_init = f_init
        self.f_next = f_next
        self.options = options
        self.word_dict = word_dicts[0]
        self.word_dict_trg = word_dict_trg
        self.word_idict_trg = word_idict_trg

        
    def src_sentence2id(self, sentence):
        return [[self.word_dict[w] for w in sentence.strip().split()]]


    def trg_id2sentence(self, id_list):
        ww = []
        for w in id_list:
            if w == 0:
                break
            ww.append(self.word_idict_trg[w])
        return ' '.join(ww)


    def set_source_sentence(self, sentence):
        seq = self.src_sentence2id(sentence)
        self.source_sentence = numpy.array(seq).T.reshape([len(seq[0]), len(seq), 1])

    def init_for_graph(self):
        next_state, ctx0 = self.f_init(self.source_sentence)
        next_w = -1 * numpy.ones((1,)).astype('int64')  # bos indicator
        return next_state, ctx0, next_w


    def score_arc(self, word, S=None):
        '''
        Given word and state, return P(word|state) and new state
        The state is a tuple (NMT_state, NMT_context, previous_word)
        '''
        if S == None:
            S = self.init_for_graph()

        nmt_state, nmt_context, prev_word = S
        ctx = numpy.tile(nmt_context, [1, 1])
        inps = [prev_word, ctx, nmt_state]
        probdist, word_prediction, nmt_state_next = self.f_next(*inps)
        word2 = numpy.array([word]).astype('int64')
        prob = probdist[0][word]
        return prob, (nmt_state_next, nmt_context, word2)



if __name__ == "__main__":
    model='/home/hltcoe/kduh/p/mt/nmt_kftt/model/ja5000-en5000/exp1-400-100.npz'
    scorer = ArcScorer(model)
    seq = scorer.src_sentence2id("曹@@ 洞 宗 の 開祖 。")
    print "Input", seq
    scorer.set_source_sentence("曹@@ 洞 宗 の 開祖 。")


    sample, score, word_probs, alignment = gen_sample([scorer.f_init], [scorer.f_next], numpy.array(seq).T.reshape([len(seq[0]), len(seq), 1]), trng=trng, k=1, maxlen=200, stochastic=False, argmax=False, return_alignment=False, suppress_unk=True)



    print "Output", sample[0]
    print scorer.trg_id2sentence(sample[0])

    def emulate(output_sequence):
        ss=None
        pps = []
        for i in range(len(output_sequence)):
            ww = output_sequence[i]
            pp,ss = scorer.score_arc(ww,S=ss)
            pps.append(pp)
        return pps

    print emulate(sample[0])
    print word_probs

