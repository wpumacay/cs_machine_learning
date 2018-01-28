## BASED ON Conchylicultor's implementation
## see https://github.com/Conchylicultor/DeepQA

import numpy as np
import tensorflow as tf
from config import ModelConfig, DataConfig

class ProjectionOp :

    """ Single layer perceptron
    Project input tensor on the output dimension
    """

    def __init__( self, shape, scope=None, dtype=None ):

        """
        Args:
            shape: a tuple (input dim, output dim)
            scope (str): encapsulate variables
            dtype: the weights type
        """

        assert len( shape ) == 2

        self.scope = scope

        with tf.variable_scope( 'weights_' + self.scope ) :

            self.W_t = tf.get_variable( 'weights', shape, dtype = dtype )
            self.b = tf.get_variable( 'bias', shape[0],
                                      initializer=tf.constant_initializer(),
                                      dtype = dtype )
            self.W = tf.transpose( self.W_t )

    def getWeights( self ):
        """ Convenience method for some tf arguments
        """
        return self.W, self.b

    def __call__( self, X ) :
        """ Project the output of the decoder into the vocabulary space
        Args:
            X (tf.Tensor): input value
        """
        with tf.name_scope( self.scope ):
            return tf.matmul( X, self.W ) + self.b



class Seq2SeqModel :


    def __init__( self, textData ) :

        self.m_textData = textData

        self.m_encoderInputs = None
        self.m_decoderInputs = None
        self.m_decoderTargets = None
        self.m_decoderWeights = None

        self.m_decoderOutputs = None
        self.m_states = None

        self.m_lossFcn = None
        self.m_optimizer = None
        self.m_outputs = None

        self.m_encDecCells = None

        print( 'building model' )
        self._buildNetwork()
        print( 'done' )

    def _buildNetwork( self ) :

        # create the placeholders

        with tf.name_scope( 'model_encoder' ) :
            # encoder inputs accept the one-hot encoding of the word
            self.m_encoderInputs = [ tf.placeholder( tf.int32, [ None, ], name = "model_encoder_inputs" ) 
                                    for _ in range( ModelConfig.INPUT_SEQUENCE_LENGTH )  ]

        with tf.name_scope( 'mode_decoder' ) :
            self.m_decoderInputs  = [ tf.placeholder( tf.int32,   [ None, ], name = 'model_decoder_inputs' ) for _ in range( ModelConfig.OUTPUT_SEQUENCE_LENGTH ) ]
            self.m_decoderTargets = [ tf.placeholder( tf.int32,   [ None, ], name = 'model_decoder_targets' ) for _ in range( ModelConfig.OUTPUT_SEQUENCE_LENGTH ) ]
            self.m_decoderWeights = [ tf.placeholder( tf.float32, [ None, ], name = 'model_decoder_weights' ) for _ in range( ModelConfig.OUTPUT_SEQUENCE_LENGTH ) ]

        # Create the layers structure of the rnn encoder and decoder
        _cells = []

        for _ in range( ModelConfig.NUM_LAYERS ) :

            _encDecCell = tf.contrib.rnn.BasicRNNCell( ModelConfig.NUM_HIDDEN_UNITS )

            if ModelConfig.CELL_TYPE == 'LSTM' :
                _encDecCell = tf.contrib.rnn.BasicLSTMCell( ModelConfig.NUM_HIDDEN_UNITS )
            elif ModelConfig.CELL_TYPE == 'GRU' :
                _encDecCell = tf.contrib.rnn.GRUCell( ModelConfig.NUM_HIDDEN_UNITS )

            # if in training, add dropout
            # see for reference : http://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf
            if not ModelConfig.IS_TEST :
                _encDecCell = tf.contrib.rnn.DropoutWrapper( _encDecCell,
                                                             input_keep_prob = 1.0,
                                                             output_keep_prob = ModelConfig.DROPOUT )

            _cells.append( _encDecCell )

        # using sampled softmax
        _outputProjection = None
        if 0 < ModelConfig.SOFTMAX_SAMPLES and ModelConfig.SOFTMAX_SAMPLES < self.m_textData.getVocabularySize() :
            _outputProjection = ProjectionOp( ( self.m_textData.getVocabularySize(), ModelConfig.NUM_HIDDEN_UNITS ),
                                              scope = 'model_softmax_projection',
                                              dtype = tf.float32 )

            def sampledSoftmax( labels, logits ) :
                labels = tf.reshape( labels, [-1, 1] )  # Add one dimension (nb of true classes, here 1)

                # We need to compute the sampled_softmax_loss using 32bit floats to
                # avoid numerical instabilities.
                localWt     = tf.cast( _outputProjection.W_t, tf.float32 )
                localB      = tf.cast( _outputProjection.b, tf.float32 )
                localInputs = tf.cast( logits, tf.float32 )

                return tf.cast( tf.nn.sampled_softmax_loss( localWt,  # Should have shape [num_classes, dim]
                                                            localB,
                                                            labels,
                                                            localInputs,
                                                            ModelConfig.SOFTMAX_SAMPLES,  # The number of classes to randomly sample per batch
                                                            self.m_textData.getVocabularySize() ),  # The number of classes
                                                            tf.float32 )

        # see https://r2rt.com/recurrent-neural-networks-in-tensorflow-ii.html
        # algo, check MultiRNNCell in rnn_cell_impl.py in the tensorflow code ( call nethod at line 1202 )
        self.m_encDecCells = tf.contrib.rnn.MultiRNNCell( _cells )

        if ModelConfig.USE_ATTENTION :
            self.m_decoderOutputs, self.m_states = tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                                                        self.m_encoderInputs,
                                                        self.m_decoderInputs,
                                                        self.m_encDecCells,
                                                        self.m_textData.getVocabularySize(),
                                                        self.m_textData.getVocabularySize(),
                                                        embedding_size = ModelConfig.EMBEDDINGS_SIZE,
                                                        output_projection = _outputProjection.getWeights() if _outputProjection else None,
                                                        feed_previous = ModelConfig.IS_TEST )
        else :
            self.m_decoderOutputs, self.m_states = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
                                                        self.m_encoderInputs,
                                                        self.m_decoderInputs,
                                                        self.m_encDecCells,
                                                        self.m_textData.getVocabularySize(),
                                                        self.m_textData.getVocabularySize(),
                                                        embedding_size = ModelConfig.EMBEDDINGS_SIZE,
                                                        output_projection = _outputProjection.getWeights() if _outputProjection else None,
                                                        feed_previous = ModelConfig.IS_TEST )


        if ModelConfig.IS_TEST :
            if not _outputProjection :
                self.m_outputs = self.m_decoderOutputs
            else :
                self.m_outputs = [ _outputProjection( _decOutput ) for _decOutput in self.m_decoderOutputs ]

        else :

            self.m_lossFcn = tf.contrib.legacy_seq2seq.sequence_loss(
                                    self.m_decoderOutputs,
                                    self.m_decoderTargets,
                                    self.m_decoderWeights,
                                    self.m_textData.getVocabularySize(),
                                    softmax_loss_function = sampledSoftmax if _outputProjection else None )
            tf.summary.scalar( 'loss', self.m_lossFcn )

            _opt = tf.train.AdamOptimizer(
                                learning_rate = ModelConfig.LEARNING_RATE,
                                beta1 = 0.9,
                                beta2 = 0.999,
                                epsilon = 1e-08 )

            self.m_optimizer = _opt.minimize( self.m_lossFcn )

    def step( self, batch ) :

        # batch input is an object containing :
        #     * testmode  : just input data ( input-> encoder seq, decoder seq, target seq, weights )
        #     * trainmode : input and target output
        # returns the operators to run the session

        _feedDict = {}
        _ops = None

        if not ModelConfig.IS_TEST : # training mode

            for i in range( ModelConfig.INPUT_SEQUENCE_LENGTH ) :
                _feedDict[ self.m_encoderInputs[ i ] ] = batch['encoderSeqs'][ i ]

            for i in range( ModelConfig.OUTPUT_SEQUENCE_LENGTH ) :
                _feedDict[ self.m_decoderInputs[ i ] ] = batch['decoderSeqs'][ i ]
                _feedDict[ self.m_decoderTargets[ i ] ] = batch['targetSeqs'][ i ]
                _feedDict[ self.m_decoderWeights[ i ] ] = batch['weights'][ i ]

            ops = ( self.m_optimizer, self.m_lossFcn )

        else :

            for i in range( ModelConfig.INPUT_SEQUENCE_LENGTH ) :
                _feedDict[ self.m_encoderInputs[ i ] ] = batch['encoderSeqs'][ i ]

            _feedDict[ self.m_decoderInputs[0] ] = [ self.m_textData.m_vocab[ DataConfig.TOKEN_START ] ]

            ops = ( self.m_outputs, )

        return ops, _feedDict