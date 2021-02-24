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

class Tokenizer():
    """
    Tokenization class.

    A model that intakes strings, and converts the text

    Args:
    ::param (str, defaults to 'en'); the language to tokenize in.
    """
    def __init__(self, lang='en'):

        if lang == 'en':
            self.nlp = spacy.load('en', disable=['parser', 'ner'])
            self.stop_words = self.nlp.Defaults.stop_words
        else:
            raise Exception('English only language supported')

    def encode(self, phrase):
        """
        Given a phrase, tokenize the text.
        Make sure character stripping is last.
        """
        # Enforce lowercase

        # Remove strange text
        text = self.omit_numbers(phrase.lower())

        # Remove numbers
        text = self.remove_stop(text, self.nlp, self.stop_words)

        # Omit Stop words
        text = self.strip_character(text)

        # Lemmatize
        text = self.lemmatize(text, self.nlp)

        return text

    @staticmethod
    def strip_character(phrase):
        """
        Omit non-alphanumeric characters for string input.

        Args:
        ::param phrase (str) - input phrase

        Returns string without special symbols
        """
        return re.sub("\W+", " ", phrase)

    @staticmethod
    def omit_numbers(phrase):
        """ Remove numbers; phrase is full string """
        return "".join([c.lower() for c in phrase if not c.isdigit()])

    @staticmethod
    def remove_stop(phrase, nlp, stop_words):
        """
        Omits stop words from phrase

        Args:
        ::param phrase (str) - input phrase
        ::param nlp (spacy.lang model) - defaults to English
        ::param stop_words (set of str) - which words to omit. May be customized.

        Returns str without stop-words
        """
        return " ".join([tok.text for tok in nlp(phrase) if tok.text not in stop_words])

    @staticmethod
    def lemmatize(phrase, nlp):
        """
        Given a str, convert to lemmatized tokens.
        Returns list of strings

        Args:
        ::param phrase (str) - input phrase
        ::param nlp (spacy.lang model) - defaults to English

        Returns tokenized list of strings

        """
        return [token.lemma_.text for token in nlp(phrase)]
