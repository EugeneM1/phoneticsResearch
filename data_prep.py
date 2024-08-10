import fasttext
import nltk
from nltk.corpus import cmudict

# just downloading the cmu dictionary
nltk.download('cmudict')
cmu_dict = cmudict.dict()

#  pre-trained vectors
fasttext_model = fasttext.load_model('cc.en.300.bin') 

def get_word_embedding(word):
    return fasttext_model.get_word_vector(word)

def get_phonetic_representation(word):
    return cmu_dict.get(word.lower())

def generate_ngrams(word, n):
    return [word[i:i+n] for i in range(len(word)-n+1)]

