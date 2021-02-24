"""
2021.02.23

natasha@mindsdb.com

English form tokenizer
to clean words in corpus.

Steps:
(1) Remove lower case
(2) Convert alphanumeric characters
(3) Strip weird tokens (*** or such)
    -> hyphenated words are partnered.
(4) Lemmatize
(5) Stem words

TODO:
- Misspellings
- Make my own num2word; not great to have package dependency
- look into part of speech tagging
- strip_character extra white space nonsense; just double check

# Lemmatizing may be slow

"""
import re
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

import sys

#try:
#    nlp = spacy.load('en', disable=['parser', 'ner'])
#except LookupError:
#    !{sys.executable} -m spacy download en ## FIXME This is pretty hideous
#    nlp = spacy.load('en', disable=['parser', 'ner'])


nlp = spacy.load('en', disable=['parser', 'ner'])
stop_words = nlp.Defaults.stop_words

def strip_character(phrase):
    """
    Omit non-alphanumeric characters for string input.

    Preserves spaces in text and keeps number
    """
    return re.sub("\W+", " ", phrase)


def omit_numbers(phrase):
    """ Remove numbers; phrase is full string """
    return "".join([c.lower() for c in phrase if not c.isdigit()])


def remove_stop(phrase):
    """ Omits stop words from phrase """
    return " ".join([tok.text for tok in nlp(phrase) if tok.text not in stop_words])


def lemmatize(phrase):
    """
    Given a str, convert to lemmatized tokens.
    Returns list of strings
    """
    return [token.lemma_ for token in nlp(phrase)]


def tokenize(phrase):
    """
    Given a phrase, tokenize the text.
    Make sure character stripping is last.
    """
    # Enforce lowercase

    # Remove strange text
    text = omit_numbers(phrase.lower())

    # Remove numbers
    text = remove_stop(text)

    # Omit Stop words
    text = strip_character(text)

    # Lemmatize
    text = lemmatize(text)

    return text