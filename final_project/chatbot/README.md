

# Making a chatbot using the Seq2Seq model

This is a simple implementation of the [**sequence to sequence model**](https://arxiv.org/abs/1409.3215) to make a chatbot (based on the paper [**"A Neural conversational model"**](https://arxiv.org/abs/1506.05869) by Oriol Vinyals and Quoc Le.

We used tensorflow to implement the model, and based the implementation in the following references :

*   [**DeepQA**](https://github.com/Conchylicultor/DeepQA) repo by Etienne Pot
*   [**Stanford-tensorflow-tutorials**](https://github.com/chiphuyen/stanford-tensorflow-tutorials/tree/master/2017/assignments/chatbot) repo by Chip Huyen

There are 3 main sections :

*   [About the model](#section1)
*   [Implementation](#section2)
*   [Results and thoughts](#section3)

## <a name="section1"></a> About the model

![base model](_img/img_encoderDecoder.jpg)

The sequence to sequence model consists of two RNN structures: a Encoder and a Decoder.

Basically, the encoder ( implemented using a sequence to vector RNN ) is in charge of compressing the information of the sequence given into a vector that summarizes all this information. Then this vector is used by the decoder ( implemented using a vector to sequence RNN ) to generate an output sequence.

The use of RNNs for these kind of tasks is described in various sources, an basically we use them because we have to deal with an input that is a sequence of non-defined length. If we used a basic Feed forward net, we should have to specify a fixed size for the input ( maybe a huge vector that would be a concatenation of the sequence, with some kind of padding, which would be very sparse ).

Next, we will analyze each component of the model :

##Encoder

* Word embeddings :

The inputs to our RNN are a sequence of vectors, which in our case should represent each word of a given sequence of words. This is achieved by using the word-embedding of a certain word, which is a fixed size vector that represents the word in a high dimensional space.

![encoderInputs](_img/img_encoderInputs.jpg)

Because the one-hot vectors are used to lookup the word-embedding from the embeddings matrix, we can basically replace the matrix-vector multiplications by a row look-up. So, the inputs to our models are integers that will look-up the embeddings.

* Vocabulary :

As described earlier, the inputs are just look-up unique indices which will look up an embedding related to a word. We have then to basically make a big dictionary of word-LookupIndex mappings that will serve the purpose of transforming a sequence into an array of indices to feed to our encoder. A portion of this mapping is show below.

    * fawn - 38069
    * woods - 2168
    * clotted - 27995
    * \padding\ - 0
    * \unknown\ - 1
    * \start\ - 2
    * \stop\ - 3
    * hanging - 1345
    * woody - 5332
    * tingly - 47470
    * localized - 27996
    * spidery - 27997
    * sevens - 24130
    * disobeying - 24131
    * mutinied - 38070
    * mathison - 31647

Here, each word is given an id, and as you can see there are some  symbols that are given special ids: \padding\, \unknown\, \start\, \stop\, being used to pad a sequence ( complete up to size ), replace in case not found in dictionary, start a sequence for decoder and stop a sequence for the decoder, respectively.

* Input sequence length :

We could keep feeding the encoder data, and it would happily keep encoding this data into the final vector, but the implementation we based on uses a fixed-size sequence length. This serves like the number of iterations in time that our RNN will compute. It also kind of helps when dealing with long sequences, as if some information could be split into two parts that have differente context, then the relationship between these wouldn't be mixed into a one vector, but two.

Of course, there are implementations that use dynamic sequence length, as described [**here**](https://danijar.com/variable-sequence-lengths-in-tensorflow/) and [**here**](https://r2rt.com/recurrent-neural-networks-in-tensorflow-iii-variable-length-sequences.html), or tensorflow's [**dynamic_rnn**](https://www.tensorflow.org/versions/master/api_docs/python/tf/nn/dynamic_rnn)

* Padding :

Some sequences may be shorter than the input-sequence, so we use padding to make this fit into the encoder's input size. Basically, if the sentence "hey bro, how are you?" is split into "[ 'hey', 'bro', ',', 'how', 'are', 'you', '?' ]", and then converted into "[ 10, 25, 75, 454, 123, 657, 1087 ]" we can pad it to a size of 10 by adding the remaining slots as '0' ( or the padding index of the padding symbol in our vocabulary ), which will become "[ 10, 25, 75, 454, 123, 657, 1087, 0, 0, 0 ]"

![encoderPadding](_img/img_encoderPadding.jpg)

##RNN cells :

There are some options as to what kind of cell to use in our RNN, which could be a vanilla RNN cell, an LSTM cell and a GRU cell.

Recall, a RNN can be seen in an unrolled way, as show in the following picture :

![rnnUnrolled](_img/img_rnnUnrolled.jpg)

Each of those rectangles with an 'A' represent an RNN cell ( here, just to be clear, when they mean single cell, they mean this is a full layer of this computing cells  )

The vanilla RNNs have as single computing unit the following RNN cell :

![rnnVanillaCell](_img/img_rnnVanillaCell.jpg)

An its computation is given by the following :

![rnnVanillaComputation](https://latex.codecogs.com/gif.latex?h_{t}&space;=&space;tanh(&space;W_{x}&space;x_{t}&space;&plus;&space;W_{h}&space;h_{t-1}))

The cells we used are LSTMs and GRUs, both part of the tensorflow API. Both these cells have the advantage of filtering some of the input from previous states so that we avoid the vanishing gradient problem because of constant multiplications. Instead, they update with sums, which generates a nicer gradient.

For more info about these cells, you could check [**this**](http://www.wildml.com/2015/10/recurrent-neural-network-tutorial-part-4-implementing-a-grulstm-rnn-with-python-and-theano/) and [**this**](https://theneuralperspective.com/2016/11/17/recurrent-neural-network-rnn-part-4-custom-cells/)

An LSTM cell looks like this :

![rnnLSTMCell](_img/img_rnnLstmCell.jpg)

And a GRU cell looks like this :

![rnnLSTMCell](_img/img_rnnGruCell.jpg)

GRU cells are quite similar in performance to LSTM cells, but their upside is that they perform less computation that LSTMs.

##Decoder

Once the thought vector is generated in the last iteration of the encoder we can use it in the decoder to generate the output sequence. The thought vector is fed as the first hidden state of the decoder cell and a start token is fed as first input to the same cell. 

The model is a bit different depending in the mode chosen : training or testing, which we will describe below.

* Training Mode

In this mode, the **subsequent inputs** ( after the start token ) to the decoder cell are the **target outputs** that we want to predict. For example, if the answer to the encoder sequence 'hey man' was 'hello bro', then the 'hello' target is fed as the input in the second step of the decoder.



* Testing mode

In this mode, we basically need the model to generate the whole sequence on its own, so the inputs used for the decoder in subsequent iterations are the previous predicted outputs of the decoder. So, for example, if the input to the encoder was 'hey man', and our first predicted output word was 'hi', then this 'hi' is fed into the next input to the decoder.


## <a name="section2"></a> Implementation Details

We used tensorflow and its API to construct our model. This can be found in the [modelSeq2Seq.py](ml/modelSeq2Seq.py) file, the data utility can be found in [dataUtils.py](ml/dataUtils.py), the chatbot entrypoint in the [chatbot.py](chatbot.py) file, and the configuration parameters can be found in the [config.py](ml/config.py) file.

### Model implementation

We first create some placeholders according to the encoder-decoder sequence sizes.

```python
        with tf.name_scope( 'model_encoder' ) :
            # encoder inputs accept the one-hot encoding of the word
            self.m_encoderInputs = [ tf.placeholder( tf.int32, [ None, ], name = "model_encoder_inputs" ) 
                                    for _ in range( ModelConfig.INPUT_SEQUENCE_LENGTH )  ]

        with tf.name_scope( 'mode_decoder' ) :
            self.m_decoderInputs  = [ tf.placeholder( tf.int32,   [ None, ], name = 'model_decoder_inputs' ) for _ in range( ModelConfig.OUTPUT_SEQUENCE_LENGTH ) ]
            self.m_decoderTargets = [ tf.placeholder( tf.int32,   [ None, ], name = 'model_decoder_targets' ) for _ in range( ModelConfig.OUTPUT_SEQUENCE_LENGTH ) ]
            self.m_decoderWeights = [ tf.placeholder( tf.float32, [ None, ], name = 'model_decoder_weights' ) for _ in range( ModelConfig.OUTPUT_SEQUENCE_LENGTH ) ]
```

As the names describe, the are creating the inputs to both encoder and decoder, the target outputs of the decoder ( used in training mode ) and the masks of the decoder inputs. This is used to deal when padding tokens are fed as inputs, as we make them 0 when padding tokens are used.

We then create the cells itself.

```python
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

        # see https://r2rt.com/recurrent-neural-networks-in-tensorflow-ii.html
        # algo, check MultiRNNCell in rnn_cell_impl.py in the tensorflow code ( call nethod at line 1202 )
        self.m_encDecCells = tf.contrib.rnn.MultiRNNCell( _cells )
```

With the default configuration, we create a 2-layer RNN with 512 hidden units for each cell, which uses LSTM cells.

We then create the seq2seq model itself, using the tensorflow API :

```python
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
```

If using the attention mechanism, we create the model using tf's **embedding_attention_seq2seq**. If not, we create the model with **embedding_rnn_seq2seq**. 

As the name suggests, the model is created with embeddings in mind. It creates an embedding look-up layer in between both encoder inputs and decoder inputs. You can check this in tensorflows's source code :

```
### tf's insides of embedding_rnn_seq2seq

## encoder creation snippet
    
    # ...

    # Encoder.
    encoder_cell = copy.deepcopy(cell)
    encoder_cell = core_rnn_cell.EmbeddingWrapper(
        encoder_cell,
        embedding_classes=num_encoder_symbols,
        embedding_size=embedding_size)
    _, encoder_state = rnn.static_rnn(encoder_cell, encoder_inputs, dtype=dtype)

    # ...

## decoder creation snippet

    # ...

    # Decoder.
    if output_projection is None:
      cell = core_rnn_cell.OutputProjectionWrapper(cell, num_decoder_symbols)

    if isinstance(feed_previous, bool):
      return embedding_rnn_decoder(
          decoder_inputs,
          encoder_state,
          cell,
          num_decoder_symbols,
          embedding_size,
          output_projection=output_projection,
          feed_previous=feed_previous)

    # If feed_previous is a Tensor, we construct 2 graphs and use cond.
    def decoder(feed_previous_bool):
      reuse = None if feed_previous_bool else True
      with variable_scope.variable_scope(
          variable_scope.get_variable_scope(), reuse=reuse):
        outputs, state = embedding_rnn_decoder(
            decoder_inputs,
            encoder_state,
            cell,
            num_decoder_symbols,
            embedding_size,
            output_projection=output_projection,
            feed_previous=feed_previous_bool,
            update_embedding_for_previous=False)
        state_list = [state]
        if nest.is_sequence(state):
          state_list = nest.flatten(state)
        return outputs + state_list

    # ...

```

As you can, there are some calls to some functions that will add the embeddings part into the computation graph.

Also, it can be noticed that there is quite some code that wrapps some functionality when building the computation graph. Keep in mind that to use tensorflow you have to make a computation graph, so most of the time you would just see wrappers around wrappers that build this computation. Only in the last unwrapper-parts of the code you could see the actual computations made ( activations, multiplications, concats, etc. ). If in doubt about the implementation, just follow along some of the tf's source code. This will help you to see the actual computations made and check that with the theory.

We then create the final parts of the model, namely the actual outputs and the loss function.

```python
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
```

Here, we check if whether we are in test mode or not. If so, we create only the outputs using the output-projection trick ( if used ). This is used if the loss function we use is [**sampled softmax**](https://www.tensorflow.org/versions/r1.0/tutorials/seq2seq#sampled_softmax_and_output_projection). You can check [**this**](https://stackoverflow.com/questions/39573188/output-projection-in-seq2seq-model-tensorflow) for some more insight.

Finally, the last part of the model class deals with running the model given a batch of data.

```python

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

```

Here, we create the parameters to run the session on, and the dictionary of parameters to feed to the session. If in training mode, we feed encoderInputs ( input sequence ), decoderInputs ( targets words shifted one to the right, and with the start token in the start of the sequence ), decoderTargets ( words we want to predict ) and decoderWeights ( masks to use if deadling with padded tokens ).

## <a name="section3"></a> Results and thoughts


