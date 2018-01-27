


class ModelConfig :

    IS_TEST = False

    CELL_TYPE = 'LSTM'
    NUM_LAYERS = 2
    NUM_HIDDEN_UNITS = 512
    INPUT_SEQUENCE_LENGTH = 10
    OUTPUT_SEQUENCE_LENGTH = 10
    USE_ATTENTION = False

    DROPOUT = 0.9

    EMBEDDINGS_SIZE = 64

    SOFTMAX_SAMPLES = 512
    LEARNING_RATE = 0.002

    BATCH_SIZE = 256

    NUM_EPOCHS = 30

class DataConfig :

    DATA_PATH = 'data/'
    CURRENT_DATASET_ID = "cornell-dataset"
    
    TOKEN_PAD = '<pad>'
    TOKEN_UNKNOWN = '<unk>'
    TOKEN_START = '<s>'
    TOKEN_END = '<\s>'

    MIN_COUNT_THRESHOLD = 5
    USE_PREPROCESSED = True