from wordfreq import zipf_frequency
import spacy
import os.path
import enchant

BASE = os.path.dirname(os.path.abspath(__file__))
nlp = spacy.load("en_core_web_lg")
dictionary = enchant.Dict("en_GB")


def get_word_frequency_number(token):
    return zipf_frequency(token.text, 'en')


def search_word(word):
     return dictionary.check(word)



