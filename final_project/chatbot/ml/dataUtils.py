

import os
import random
import re
import numpy as np

from tqdm import tqdm

from config import DataConfig, ModelConfig


class DataUtils :

    def __init__( self, datasetId ) :

        self.m_datasetId = datasetId

        self.m_encLines = None
        self.m_decLines = None
        self.m_vocab = {}
        self.m_invVocab = {}

        self.m_encSeqs = []
        self.m_decSeqs = []

        if self.m_datasetId == 'cornell-dataset' :
            print( 'parsing cornell-dataset' )
            self._parseCornellDataset()
        else :
            print( 'wip: adding other datasets' )

    def dumpVocab( self ) :

        assert( len( self.m_vocab ) > 0 )

        with open( 'dumpVocab.txt', 'w' ) as _fileHandle :
            for _word in self.m_vocab :

                _fileHandle.write( _word + ' - ' + str( self.m_vocab[ _word ] ) + '\n' )

    def dumpSampleBatch( self ) :

        assert( len( self.m_vocab ) > 0 )

        _bEncInputs, _bDecInputs, _bDecTargets, _bMasks = self.getBatch( ModelConfig.BATCH_SIZE )

        with open( 'dumpBatch.txt', 'w' ) as _fileHandle :

            for i in range( len( _bEncInputs ) ) :
                _fileHandle.write( str( _bEncInputs[i] ) + '\n' )

            _fileHandle.write( '-----------------' )

            for i in range( len( _bDecInputs ) ) :
                _fileHandle.write( str( _bDecInputs[i] ) + '\n' )

            _fileHandle.write( '-----------------' )

            for i in range( len( _bDecTargets ) ) :
                _fileHandle.write( str( _bDecTargets[i] ) + '\n' )

            _fileHandle.write( '-----------------' )

            for i in range( len( _bMasks ) ) :
                _fileHandle.write( str( _bMasks[i] ) + '\n' )

    def _saveDataset( self ) :

        assert( len( self.m_vocab ) > 0 )

        # save vocabulary
        _sortedVocab = sorted( self.m_vocab, key = self.m_vocab.get, reverse = False )

        with open( 'data_vocabulary.txt', 'w' ) as _fileHandle :
            for _word in _sortedVocab :

                _fileHandle.write( _word + '\n' )

        # save line ids
        with open( 'data_lineids_enc_lines.txt', 'w' ) as _fileHandle :
            for _line in self.m_encLines :

                _fileHandle.write( _line + '\n' )

        with open( 'data_lineids_dec_lines.txt', 'w' ) as _fileHandle :
            for _line in self.m_decLines :

                _fileHandle.write( _line + '\n' )

        # save line ids
        with open( 'data_lineids_enc.txt', 'w' ) as _fileHandle :
            for _seq in self.m_encSeqs :

                _fileHandle.write( ' '.join( str( _id ) for _id in _seq ) + '\n' )

        with open( 'data_lineids_dec.txt', 'w' ) as _fileHandle :
            for _seq in self.m_decSeqs :

                _fileHandle.write( ' '.join( str( _id ) for _id in _seq ) + '\n' )

    def _parseCornellDataset( self ) :

        if not DataConfig.USE_PREPROCESSED :

            _id2line = self._pcGetLines()
            _convSequences = self._pcGetConvSequences()
            self.m_encLines, self.m_decLines = self._pcGetQAs( _id2line, _convSequences )

            self._buildVocabulary( self.m_encLines, self.m_decLines )
            self.m_encSeqs, self.m_decSeqs = self._lines2ids( self.m_encLines, self.m_decLines )

            self._saveDataset()

        else :

            with open( 'preprocessed/data_vocabulary.txt', 'r' ) as _fileHandle :
                _lines = _fileHandle.readlines()

                _indx = 0
                for _line in tqdm( _lines, desc = 'parsing preprocessed vocabulary' ) :
                    self.m_vocab[ _line[ :-1 ] ] = _indx
                    _indx += 1

            self.m_invVocab = dict( zip( self.m_vocab.values(), self.m_vocab.keys() ) )

            with open( 'preprocessed/data_lineids_enc.txt', 'r' ) as _fileHandle :
                _lines = _fileHandle.readlines()

                indx = 1
                for _line in tqdm( _lines, desc = 'parsing preprocessed enc lines' ) :
                    _strIds = ( _line[:-1] ).split( ' ' )

                    # print( 'indx: ', indx )
                    _ids = [ int( _strId ) for _strId in _strIds ]

                    self.m_encSeqs.append( _ids )

                    indx += 1


            with open( 'preprocessed/data_lineids_dec.txt', 'r' ) as _fileHandle :
                _lines = _fileHandle.readlines()

                for _line in tqdm( _lines, desc = 'parsing preprocessed dec lines' ) :
                    _strIds = ( _line[:-1] ).split( ' ' )

                    _ids = [ int( _strId ) for _strId in _strIds ]

                    self.m_decSeqs.append( _ids )

    def _pcGetLines( self ) :

        # get lines into a dictionary ******************************************

        _id2line = {}
        _filePath = os.path.join( DataConfig.DATA_PATH, 'movie_lines.txt' )

        with open( _filePath, 'rb' ) as _fileHandle :

            _lines = _fileHandle.readlines()

            for _line in tqdm( _lines, desc = 'Parse lines: ' ) :

                _parts = _line.split( ' +++$+++ ' )
                # just in case one line is not in the format
                if len( _parts ) == 5 :

                    if _parts[4][-1] == '\n' :
                        ## take all except the return char
                        _parts[4] = _parts[4][:-1]

                    _id2line[ _parts[0] ] = _parts[4]

        return _id2line

    def _pcGetConvSequences( self ) :

        # Get conversation pairs ***********************************************

        _convSequences = []
        _filePath = os.path.join( DataConfig.DATA_PATH, 'movie_conversations.txt' )

        with open( _filePath, 'rb' ) as _fileHandle :

            _lines = _fileHandle.readlines()

            for _line in tqdm( _lines, desc = 'Parse conversations: ' ) :

                _parts = _line.split( ' +++$+++ ' )
                # just in case one line is not in the format
                if len( _parts ) == 4 :

                    _conversation = []

                    for _convId in _parts[3][1:-2].split( ', ' ) :
                        _conversation.append( _convId[1:-1] )

                    _convSequences.append( _conversation )

        return _convSequences


    def _pcGetQAs( self, id2line, convSequences ) :

        _questions, _answers = [], []

        for _convSeq in convSequences :

            if len( _convSeq ) < 2 :
                # just in case there is a one-line conversation
                continue

            for i in range( len( _convSeq ) - 1 ) :
                # note that one answer is the question in the next iteration

                if len( id2line[ _convSeq[ i ] ] ) < 1 :
                    continue

                if len( id2line[ _convSeq[ i + 1 ] ] ) < 1 :
                    continue

                _questions.append( id2line[ _convSeq[ i ] ] )
                _answers.append( id2line[ _convSeq[ i + 1 ] ] )

        # sanity check
        assert( len( _questions ) == len( _answers ) )

        return _questions, _answers

    def _pcTokenizer( self, line ) :

        line = re.sub( '<u>', '', line )
        line = re.sub( '</u>', '', line )
        line = re.sub( '<b>', '', line )
        line = re.sub( '\[', '', line )
        line = re.sub( '\]', '', line )
        line = re.sub( "\s\s+" , " ", line )
        _words = []

        _reWordSplit = re.compile( b"([.,!?\"'-<>:;)(])" )
        _reDigits = re.compile( r"\d" )

        for _fragment in line.strip().lower().split() :
            for _token in re.split( _reWordSplit, _fragment ) :

                if not _token :
                    continue

                _token = re.sub( _reDigits, b'#', _token )

                _words.append( _token )

        return _words

    def _buildVocabulary( self, questions, answers ) :

        _vocab = {} # a placeholder vocab to build the histogram

        print( 'Bulding vocabulary ===========')

        for _question in tqdm( questions, desc = 'Histogram build with questions: ' ) :
            for _token in self._pcTokenizer( _question ) :

                if not _token in _vocab :
                    _vocab[ _token ] = 0
                else :
                    _vocab[ _token ] += 1

        for _answer in tqdm( answers, desc = 'Histogram build with answers: ' ) :
            for _token in self._pcTokenizer( _answer ) :

                if not _token in _vocab :
                    _vocab[ _token ] = 0
                else :
                    _vocab[ _token ] += 1

        # sort the keys in the histogram from greatest to lowest
        _sortedVocab = sorted( _vocab, key = _vocab.get, reverse=True)

        self.m_vocab[ DataConfig.TOKEN_PAD ] = 0
        self.m_vocab[ DataConfig.TOKEN_UNKNOWN ] = 1
        self.m_vocab[ DataConfig.TOKEN_START ] = 2
        self.m_vocab[ DataConfig.TOKEN_END ] = 3

        self.m_invVocab[0] = DataConfig.TOKEN_PAD
        self.m_invVocab[1] = DataConfig.TOKEN_UNKNOWN
        self.m_invVocab[2] = DataConfig.TOKEN_START
        self.m_invVocab[3] = DataConfig.TOKEN_END

        _indx = 4

        for _word in _sortedVocab :

            if _word < DataConfig.MIN_COUNT_THRESHOLD :
                break

            self.m_vocab[ _word ] = _indx
            self.m_invVocab[ _indx ] = _word

            _indx += 1

        assert( len( self.m_vocab ) == len( self.m_invVocab ) )

        print( 'Done =========================')

    def _lines2ids( self, encLines, decLines ) :

        _encSeqsIds = []
        _decSeqsIds = []

        for i in tqdm( range( len( encLines ) ), desc = 'One hot encoding encoder-decoder lines' ) :
            _encSeq = self.sentence2id( encLines[i] )
            _decSeq = self.sentence2id( decLines[i] )

            if len( _encSeq ) == 0 or len( _decSeq ) == 0 :
                continue

            _encSeqsIds.append( _encSeq )

            _ids = [ self.m_vocab[ DataConfig.TOKEN_START ] ] + _decSeq + [ self.m_vocab[ DataConfig.TOKEN_END ] ]
            _decSeqsIds.append( _ids )

        return _encSeqsIds, _decSeqsIds

    def getVocabularySize( self ) :
        return len( self.m_vocab )

    def getTrainingSize( self ) :
        return len( self.m_encSeqs )

    def sentence2id( self, sentence ) :
        _sentIds = []

        for _token in self._pcTokenizer( sentence ) :
            if _token in self.m_vocab :
                _sentIds.append( self.m_vocab[ _token ] )
            else :
                _sentIds.append( self.m_vocab[ DataConfig.TOKEN_UNKNOWN ] )

        return _sentIds


    def id2sentence( self, seqids, breakAtStopToken = False ) :

        _sentence = []

        for _id in seqids :
            if breakAtStopToken and self.m_invVocab[ _id ] == DataConfig.TOKEN_END :
                break
            _sentence.append( self.m_invVocab[ _id ] )

        return ' '.join( _sentence )

    def _padSequence( self, sequence, size ) :
        if len( sequence ) > size :
            return sequence[ 0:size ]
        return sequence + [ self.m_vocab[ DataConfig.TOKEN_PAD ] ] * ( size - len( sequence ) )

    def decoderOut2ids( self, decOut ) :

        _sequenceIds = []

        # Choose the words with the highest prediction score
        for _out in decOut :

            _sequenceIds.append( np.argmax( _out ) )

        return _sequenceIds

    def getTestBatch( self, encSequence ) : # generate a single size=1 batch in the adecuate format
        
        _encoderInputs, _decoderInputs, _decoderTargets = [], [], []

        for _ in range( 1 ) :
            # get a random training example from the global data
            _encoderInput = encSequence
            _decoderInput = [ self.m_vocab[ DataConfig.TOKEN_START ] ] + [ self.m_vocab[ DataConfig.TOKEN_END ] ]
            _decoderTarget = _decoderInput[ 1: ] # these are not used in the model

            _encoderInputs.append( list( reversed( self._padSequence( _encoderInput, ModelConfig.INPUT_SEQUENCE_LENGTH ) ) ) )
            _decoderInputs.append( self._padSequence( _decoderInput, ModelConfig.OUTPUT_SEQUENCE_LENGTH ) )
            _decoderTargets.append( self._padSequence( _decoderTarget, ModelConfig.OUTPUT_SEQUENCE_LENGTH ) )

        _batchEncoderInputs = self._reshapeBatch( _encoderInputs, ModelConfig.INPUT_SEQUENCE_LENGTH, 1 )
        _batchDecoderInputs = self._reshapeBatch( _decoderInputs, ModelConfig.OUTPUT_SEQUENCE_LENGTH, 1 )
        _batchDecoderTargets = self._reshapeBatch( _decoderTargets, ModelConfig.OUTPUT_SEQUENCE_LENGTH, 1 )

        _batchMasks = []

        for length_id in range( ModelConfig.OUTPUT_SEQUENCE_LENGTH ) :

            _batchMask = np.ones( 1, dtype = np.float32 )
            for batch_id in range( 1 ):
                # we set mask to 0 if the corresponding target is a PAD symbol.
                # the corresponding decoder is decoder_input shifted by 1 forward.
                if length_id < ModelConfig.OUTPUT_SEQUENCE_LENGTH - 1 :
                    target = _decoderInputs[batch_id][length_id + 1]

                if length_id == ModelConfig.OUTPUT_SEQUENCE_LENGTH - 1 or target == self.m_vocab[ DataConfig.TOKEN_PAD ] :
                    _batchMask[batch_id] = 0.0

            _batchMasks.append( _batchMask )

        return _batchEncoderInputs, _batchDecoderInputs, _batchDecoderTargets, _batchMasks        

    # like transposing the matrix
    def _reshapeBatch( self, inputs, size, batchSize ) :
        _batchInputs = []

        for length_id in range( size ) :

            _batchInputs.append( np.array( [ inputs[ batch_id ][ length_id ]
                                        for batch_id in range( batchSize ) ], dtype = np.int32 ) )
        return _batchInputs

    def getQuestions( self ) :
        return self.m_encLines

    def getAnswers( self ) :
        return self.m_decLines

    def getBatch( self, batchSize ) :

        _encoderInputs, _decoderInputs, _decoderTargets = [], [], []

        for _ in range( batchSize ) :
            # get a random training example from the global data
            _indx = random.randint( 0, len( self.m_encSeqs ) - 1 )
            _encoderInput = self.m_encSeqs[ _indx ]
            _decoderInput = self.m_decSeqs[ _indx ]
            _decoderTarget = _decoderInput[ 1: ]

            _encoderInputs.append( list( reversed( self._padSequence( _encoderInput, ModelConfig.INPUT_SEQUENCE_LENGTH ) ) ) )
            _decoderInputs.append( self._padSequence( _decoderInput, ModelConfig.OUTPUT_SEQUENCE_LENGTH ) )
            _decoderTargets.append( self._padSequence( _decoderTarget, ModelConfig.OUTPUT_SEQUENCE_LENGTH ) )

            # if len( _encoderInputs[-1] ) != ModelConfig.INPUT_SEQUENCE_LENGTH :
            #     print( 'wtf???' )

            # if len( _decoderInputs[-1] ) != ModelConfig.OUTPUT_SEQUENCE_LENGTH :
            #     print( 'what????' )

            # if len( _decoderTargets[-1] ) != ModelConfig.OUTPUT_SEQUENCE_LENGTH :
            #     print( 'whyyyy????' )

        _batchEncoderInputs = self._reshapeBatch( _encoderInputs, ModelConfig.INPUT_SEQUENCE_LENGTH, batchSize )
        _batchDecoderInputs = self._reshapeBatch( _decoderInputs, ModelConfig.OUTPUT_SEQUENCE_LENGTH, batchSize )
        _batchDecoderTargets = self._reshapeBatch( _decoderTargets, ModelConfig.OUTPUT_SEQUENCE_LENGTH, batchSize )

        _batchMasks = []

        for length_id in range( ModelConfig.OUTPUT_SEQUENCE_LENGTH ) :

            _batchMask = np.ones( batchSize, dtype = np.float32 )
            for batch_id in range( batchSize ):
                # we set mask to 0 if the corresponding target is a PAD symbol.
                # the corresponding decoder is decoder_input shifted by 1 forward.
                if length_id < ModelConfig.OUTPUT_SEQUENCE_LENGTH - 1 :
                    target = _decoderInputs[batch_id][length_id + 1]

                if length_id == ModelConfig.OUTPUT_SEQUENCE_LENGTH - 1 or target == self.m_vocab[ DataConfig.TOKEN_PAD ] :
                    _batchMask[batch_id] = 0.0

            _batchMasks.append( _batchMask )

        return _batchEncoderInputs, _batchDecoderInputs, _batchDecoderTargets, _batchMasks
