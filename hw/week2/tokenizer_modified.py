## Save the tokenizer to useit later

from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import nltk.tag
import re

stop = stopwords.words( 'english' )
porter = PorterStemmer()

def tokenizer_modified( text ):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    text = [w for w in text.split() if w not in stop]
    # stemming
    stemmed = [porter.stem(w) for w in text]
    stemmed_tagged = nltk.tag.pos_tag( stemmed )
    # filter by type == adjective or noun
    tokenized = [ w for (w,c) in stemmed_tagged if c == 'JJ' or c == 'NN' ]
        
    return text