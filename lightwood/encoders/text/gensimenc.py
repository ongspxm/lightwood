"""
2021.02.25
natasha@mindsdb.com

A word2vec implementation, using gensim, for
text.

NOTE FOR GENSIM 4.0 CHANGE SIZE AND ITERS TO VECTOR_SIZE AND EPOCHS

https://arxiv.org/abs/1607.04606

Weird note - gen sim changed vector_size to 'size'

** NOTE Natasha her own gensim with vector_size instead of size
"""

import torch
import numpy as np
from lightwood.helpers.device import get_devices
from lightwood.encoders.encoder_base import BaseEncoder
from lightwood.logger import log
from tokenizer import Tokenizer

#from text_helpers import mean_norm, last_state #TODO

from gensim.models import (
    Word2Vec,
    FastText,
    )
class GensimText(BaseEncoder):
    """
    A class depending on gensim-based word embeddings.

    Supports word2vec, and fastText.

    Args:
    is_target (Bool) - target var or not
    model_type (str, default 'word2vec') - type of model
    embed_dim (int, 100) - embedding vector dimension
    window (int, 5) - window-width to consider neightbor words
    min_count (int, 2) - consider only tokens with min_count frequency in corpus
    workers (int, 4) - number of workers
    tokenizer (class, default None) - tokenizing function; if undefined 'None'
    sent_embedder (function, default None) - sentence embedding function
    seed (int, default 1234) - model initialization seed
    epochs (int, default 5) - iterations to train
    """
    def __init__(self,
                 is_target=False,
                 model_type="word2vec",
                 embed_dim=100,
                 window=5,
                 min_count=2,
                 workers=4,
                 tokenizer=None,
                 sent_embedder=None,
                 seed=1234,
                 epochs=100,
                 ):
        super().__init__(is_target)

        self.name = model_type + " text encoder"

        self._epochs = epochs

        self._embed_dim = embed_dim
        self._window = window
        self._min_count = min_count

        if model_type == "fasttext":
            self._model = FastText(size=self._embed_dim,
                                   window=self._window,
                                   min_count=self._min_count,
                                   seed=seed,
                                   iter=self._epochs)
            #self.model_type = model_type
        else:
            self._model = Word2Vec(size=self._embed_dim,
                                   window=self._window,
                                   min_count=self._min_count,
                                   workers=workers,
                                   seed=seed,
                                   iter=self._epochs)
            #self.model_type = model_type

        # Type of sentence embedding
        if sent_embedder is None:
            self._sent_embedder = self._mean_norm
        else:
            self._sent_embedder = sent_embedder

        self._tokenizer = tokenizer

        #self.device, _ = get_devices() # Probably not useful as no torch training.

    def to(self, device, available_devices):
        """ Set torch device to CPU/CUDA """
        self._model = self._model.to(self.device)

        if self._head is not None:
            self._head = self._head.to(self.device)

        return self

    def prepare(self, priming_data, training_data=None):
        """
        Prepare the text encoder to convert text -> feature vector

        priming_data (list) list of str data
        """
        if self._prepared:
            raise Exception("You can only prepare the encoder once. For more training use train()")

        if self._tokenizer is None:
            self._tokenizer = Tokenizer()

        # Tokenize the sentences to train the model
        tokens = [self._tokenizer.encode(phrase) for phrase in priming_data]
        self._corpus_count = len(tokens)

        # Train the model
        self._model.build_vocab(tokens, progress_per=1000)
        self._model.train(sentences=tokens,
                          epochs=self._epochs,
                          total_examples=self._corpus_count)

        self._prepared = True
        self._trained_epochs = self._epochs

    def train(self, tokens, Nepochs, checktokens=True):
        """
        Train gensim model for Nepochs with the tokens input.
        If a token is NOT in the vocabulary, then.

        tokens (list of lists of str);
        Nepochs (int); number of epochs to train
        checktokens (Bool); Check for unknown tokens
        """
        if self._prepared is None:
            raise Exception("You need to call the prepare command first")

        self._trained_epochs += self._epochs

        self._model.train(sentences=tokens,
                          epochs=Nepochs,
                          total_examples=len(tokens))

    def encode(self, column_data):
        """
        Encode a sentence or sentence in column data.
        Given each sentence can have variable tokens.

        Column_data (list) list of col data from text. Will tokenize

        Returns np.array of size Nsents x Nembed
        """
        V = set(self._model.wv.vocab.keys())
        outputs = []
        for text in column_data:
            if text is None:
                text = ''

            output = self.embed(text, V)
            output = self._sent_embedder(output)
            outputs.append(output)

        return np.vstack(outputs)

    def embed(self, phrase, V):
        """
        Given a phrase, tokenizes and yields a model value

        phrase - (str) a single text instance
        V - (set) set of vocabulary words

        Returns averaged vector; 1 x Nembed
        """
        token = self._tokenizer.encode(phrase)

        # Return word vectors for each token; outputs Nembed x Ntokens
        output = np.vstack([self._model.wv[word] for word in token if word in V]).T
        return output

    def decode(self, encoded_values_tensor, max_length=100):
        raise Exception("Decoder not implemented yet.")

    @staticmethod
    def _mean_norm(xinp, dim=1):
        """
        Calculates a 1 x N_embed vector by averaging all token embeddings

        Args:
        xinp ::np array; Assumes order Nembedding x Ntokens
        dim ::int; dimension to average on
        """
        return np.mean(xinp, axis=dim)
