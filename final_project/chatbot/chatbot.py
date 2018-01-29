

import os
import sys
import math
import random
import numpy as np
import tensorflow as tf

from tqdm import tqdm

from ml import modelSeq2Seq, dataUtils, config

def _checkForRestoration( session, saver ) :

    ckpt = tf.train.get_checkpoint_state( 'checkpoints/' )
    if ckpt and ckpt.model_checkpoint_path :
        print( "Loading parameters for the Chatbot" )
        saver.restore( session, ckpt.model_checkpoint_path )
    else:
        print("Initializing fresh parameters for the Chatbot")

def testData() :

    _dataHandler = dataUtils.DataUtils( config.DataConfig.CURRENT_DATASET_ID )

    # dump dictionary to a file
    _dataHandler.dumpVocab()

    # get a batch and show it
    _dataHandler.dumpSampleBatch()

def _get_user_input():
    """ Get user's input, which will be transformed into encoder input later """
    print( "> " )
    sys.stdout.flush()
    return sys.stdin.readline()

# to test it, make sure you are parsing the data from scratch ( USE_PREPROCESSED = False in config file )
def testChatbotHardMemory() :

    _dataHandler = dataUtils.DataUtils( config.DataConfig.CURRENT_DATASET_ID )

    _model = modelSeq2Seq.Seq2SeqModel( _dataHandler )    

    _saver = tf.train.Saver()

    with tf.Session() as sess :

        sess.run( tf.global_variables_initializer() )
        _checkForRestoration( sess, _saver )

        _questions = _dataHandler.getQuestions()
        _answers = _dataHandler.getAnswers()

        TEST_SIZE = len( _questions )
        TEST_SAMPLE_SIZE = 100

        _fileHandle = open( 'testHardMemory.txt', 'w' )

        for i in tqdm( range( TEST_SAMPLE_SIZE ), desc = 'testHardMemory' ) :

            _indx = random.randint( 0, TEST_SIZE - 1 )

            _line = _questions[ _indx ]
            _answer = _answers[ _indx ]

            _sentIds = _dataHandler.sentence2id( str( _line ) )

            if ( len( _sentIds ) > config.ModelConfig.INPUT_SEQUENCE_LENGTH ) :
                continue

            _encIns, _decIns, _decTargets, _decMasks = _dataHandler.getTestBatch( _sentIds )

            _batch = {}
            _batch['encoderSeqs'] = _encIns
            _batch['decoderSeqs'] = _decIns
            _batch['targetSeqs'] = _decTargets
            _batch['weights'] = _decMasks

            _ops, _feedDict = _model.step( _batch )
            _output = sess.run( _ops[0], _feedDict )

            _ids = _dataHandler.decoderOut2ids( _output )

            _response = _dataHandler.id2sentence( _ids, True )

            _fileHandle.write( 'question: ' + _line + '\n' )
            _fileHandle.write( 'answerHuman: ' + _answer + '\n' )
            _fileHandle.write( 'answerBot: ' + _response + '\n' )
            _fileHandle.write( '-------\n' )

        _fileHandle.close()

def testChatbot() :

    _dataHandler = dataUtils.DataUtils( config.DataConfig.CURRENT_DATASET_ID )

    _model = modelSeq2Seq.Seq2SeqModel( _dataHandler )    

    _saver = tf.train.Saver()

    with tf.Session() as sess :

        sess.run( tf.global_variables_initializer() )
        _checkForRestoration( sess, _saver )

        print( 'Chatbot-seq2seq ############################' )

        while True :

            _line = _get_user_input()

            if len( _line ) > 0 and _line[-1] == '\n':
                _line = _line[:-1]

            if _line == '':
                break

            _sentIds = _dataHandler.sentence2id( str( _line ) )

            if ( len( _sentIds ) > config.ModelConfig.INPUT_SEQUENCE_LENGTH ) :
                print( 'sentence too long :(' )
                continue

            _encIns, _decIns, _decTargets, _decMasks = _dataHandler.getTestBatch( _sentIds )

            _batch = {}
            _batch['encoderSeqs'] = _encIns
            _batch['decoderSeqs'] = _decIns
            _batch['targetSeqs'] = _decTargets
            _batch['weights'] = _decMasks

            _ops, _feedDict = _model.step( _batch )
            _output = sess.run( _ops[0], _feedDict )

            _ids = _dataHandler.decoderOut2ids( _output )

            _response = _dataHandler.id2sentence( _ids, True )

            print( 'answer> ', _response )

def trainChatbot() :

    _dataHandler = dataUtils.DataUtils( config.DataConfig.CURRENT_DATASET_ID )

    _model = modelSeq2Seq.Seq2SeqModel( _dataHandler )

    _saver = tf.train.Saver()

    with tf.Session() as sess :
        print( 'Starting training' )
        sess.run( tf.global_variables_initializer() )

        _iteration = 0
        _totalLoss = 0

        _globalStep = 0

        for e in range( config.ModelConfig.NUM_EPOCHS ) :

            print( "----- Epoch {}/{} ; (lr={}) -----".format( e + 1, config.ModelConfig.NUM_EPOCHS, config.ModelConfig.LEARNING_RATE ) )

            _nBatches = _dataHandler.getTrainingSize() / config.ModelConfig.BATCH_SIZE
            print( '_nBatches: ', _nBatches )

            for b in tqdm( range( _nBatches ), desc = 'training: ' ) :

                _encIns, _decIns, _decTargets, _decMasks = _dataHandler.getBatch( config.ModelConfig.BATCH_SIZE )

                _batch = {}
                _batch['encoderSeqs'] = _encIns
                _batch['decoderSeqs'] = _decIns
                _batch['targetSeqs'] = _decTargets
                _batch['weights'] = _decMasks

                _ops, _feedDict = _model.step( _batch )
                _, _loss = sess.run( _ops, _feedDict )

                _totalLoss += _loss
                _iteration += 1
                _globalStep += 1

                if _iteration % 100 == 0 :
                    _perplexity = math.exp( float( _loss ) ) if _loss < 300 else float( "inf" )
                    tqdm.write("----- Step %d -- Loss %.2f -- Perplexity %.2f" % ( _iteration, _loss, _perplexity ) )

                if _iteration % 400 == 0 :
                    print( 'saved so far :o' )
                    _saver.save( sess, 'checkpoints/chatbot', _globalStep )

            print( '_totalLoss: ', _totalLoss )

            _totalLoss = 0
            _iteration = 0



def main() :

    if config.ModelConfig.IS_TEST :
        #testChatbot()
        testChatbotHardMemory() # to test this, make sure you are parsing the data from scratch ( USE_PREPROCESSED = False in config file )
        # one thing i noticed is that the questions-answers as a pair have actually a strong relation to some previous context ...
        # by looking at the hard test, we see that it can't reproduce-memorize the human answers.
        # perhaps the dataset should be a bit different if trying to make a more open-domain bot
    else :
        trainChatbot()

if __name__ == '__main__' :

    main()