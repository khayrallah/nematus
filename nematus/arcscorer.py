#!/usr/bin/env python

import sys
import argparse
import cPickle as pkl
import json
import numpy
from theano_util import (load_params, init_theano_params)
from nmt import (build_sampler, pred_probs, build_model, prepare_data, init_params, gen_sample)
from util import load_dict, load_config
from compat import fill_options
import theano

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
trng = RandomStreams(1234)
use_noise = theano.shared(numpy.float32(0.))

class ArcScorer(object):
    
    def __init__(self, model):
        '''Loads a Nematus NMT model
        Sets the following fields in self:
        - f_init: NMT initialization function
        - f_next: NMT next function
        - word_dict: source mapping of word to id
        - word_dict_trag: target mapping of word to id
        - word_idict_trg: target mapping of id to word
        For all these, we assume there is only one version, i.e. no ensemble
        '''
        options = []
        options.append(load_config(model))
        # hacks for using old models with missing options
        fill_options(options[-1])

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

        self.nmt_state_init = None
        self.nmt_context = None

        
    def src_sentence2id(self, sentence):
        '''Convert source sentence into sequence of id's'''
        return [[self.word_dict[w]] if w in self.word_dict else [1] for w in sentence.strip().split()]


    def trg_id2sentence(self, id_list):
        '''Convert sequence of target id's into sentence'''
        ww = []
        for w in id_list:
            if w == 0:
                break
            ww.append(self.word_idict_trg[w])
        return ' '.join(ww)


    def set_source_sentence(self, sentence):
        '''This function needs to be called before running score()
        Given a source sentence, it creates the initial NMT decoder state (self.nmt_state_init) as well as the bidirectional RNN encoding of the input context (self.nmt_context) by running f_init
        '''
        seq = self.src_sentence2id(sentence) + [[0]]
        sys.stderr.write("Set NMT src sent: {} {} ({} words)\n".format(sentence, seq, len(sentence.split())))
        self.source_sentence = numpy.array(seq).T.reshape([len(seq[0]), len(seq), 1])
        self.nmt_state_init, input_rep = self.f_init(self.source_sentence)
        self.nmt_context = numpy.tile(input_rep, [1, 1])


    def score(self, state, arc):
        '''
        Given state and arc, return:
        - new state after decoding the word(s) given in arc.label
        - logProbability(word=arc.label | state, source_sentence) under the NMT model.
        Note: a state here is is a tuple (NMT_state, previously_decoded_word)
        '''

        # If state was unset, default to the initial state
        if state is None:
            bos = -1 * numpy.ones((1,)).astype('int64') # beginning of sentence indicator
            state = {'nmt_state':self.nmt_state_init, 'prev_word':bos}

        logprob = 0.0
        nmt_state, prev_word = state['nmt_state'], state['prev_word']
        words = arc.words() # there may be multiple words in an arc, so process each successively
        for word_str in words:

            # if label is epsilon, do nothing (logprob+=0); else run NMT step
            if word_str != '<eps>':

                if word_str in self.word_dict_trg:
                    word = self.word_dict_trg[word_str]
                else:
                    # NOTE: we might want to throw an exception rather than process UNK, depending on situation
                    word = self.word_dict_trg["UNK"]
                if word_str == '<eos>':
                    # NOTE: special processing for <eos>. Double-check other solution
                    word = 0 

                # run one forward step of f_next(), 
                # returns probability distribution of next word, most probable next word, and the new NMT state
                inps = [prev_word, self.nmt_context, nmt_state]
                probdist, word_prediction, nmt_state_next = self.f_next(*inps)

                # accumulate log probabilities
                logprob += numpy.log(probdist[0][word])
#                print "%s \t P(%s|...)=%f cumulativeLogProb=%f" %(arc, word_str, probdist[0][word], logprob)

                # reset NMT state and previously decoded word
                nmt_state = nmt_state_next
                prev_word = numpy.array([word]).astype('int64')

        return {'nmt_state':nmt_state, 'prev_word':prev_word}, logprob

