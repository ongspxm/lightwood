"""
2021.04.21

An example custom encoder. We will build an MLM loss-function to train a model.

This is a very contrived example to demonstrate the flexibility of the lightwood approach.

The baseclass "BaseEncoder" can be found in lightwood/encoders. This class specifically only requires a few things:  # noqa: E501

1) __init__ call; to specify whether you're training to target or not
2) prepare() call; sets the model and (optionally) trains encoder. Must intake priming data, but can be "ignored"
3) encode() call; the ability to featurize a model
4) decode() call; this is only required if the user is trying to predict in the latent space of the encoder

The script establishes a DistilBert model and trains an MLM based on the task at hand. If the task is classification, it will assign a token to each label to predict. If the task is regression, it will construct labels for each "bin" of a histogrammed approach of the numerical value.

The output of the encoder is the CLS token.

Currently the model explicitly reads "DistilBert".

Author: Natasha Seelam (natasha@mindsdb.com)
"""

# Dataset helpers
from torch.utils.data import DataLoader
from mlm_helpers import MaskedText, add_mask, create_label_tokens

# lightwood helpers
from lightwood.constants.lightwood import COLUMN_DATA_TYPES
from lightwood.helpers.device import get_devices
from lightwood.encoders.encoder_base import BaseEncoder
from lightwood.logger import log
from lightwood.helpers.torch import LightwoodAutocast

# text-model helpers
from transformers import (
    DistilBertForMaskedLM,
    DistilBertTokenizerFast,
    AdamW,
    get_linear_schedule_with_warmup,
)

# Type-hinting
from typing import List, Dict


class MLMEncoder(BaseEncoder):
    """
    An example of a custom encoder.
    Here, the model chosen will be Distilbert MLM
    Where we will train a classification task as an MLM.

    In order to build encoders, we inherit from BaseEncoder.

    Args:
    ::param is_target; data column is the target of ML.
    ::param ibatch_size; size of batch
    ::param imax_position_embeddings; max sequence length of input text
    ::param epochs; number of epochs to train model with
    ::param lr; learning-rate for model
    """

    def __init__(
        self,
        is_target: bool = False,
        batch_size: int = 10,
        max_position_embeddings: int = None,
        epochs: int = 1,
        lr: float = 1e-5,
    ):
        super().__init__(is_target)

        self.model_name = "distilbert-base-uncased"
        self.name = self.model_name + " text encoder"
        log.info(self.name)

        self._max_len = max_position_embeddings
        self._batch_size = batch_size
        self._epochs = epochs
        self._lr = lr

        # Model setup
        self._model = None
        self.model_type = None

        # Distilbert is a good balance of speed/performance hence chosen.
        self._embeddings_model_class = DistilBertForMaskedLM
        self._tokenizer_class = DistilBertTokenizerFast

        # Set the device; CUDA v. CPU
        self.device, _ = get_devices()


    def prepare(self, priming_data: List[str], training_data: Dict):
        """
        Prepare the encoder by training on the target.

        Training data must be a dict with "targets" avail.
        Automatically assumes this.

        Args:
        ::param priming_data; list of the input text
        ::param training_data; config of lightwood for encoded outputs etc.
        """
        assert (len(training_data["targets"]) == 1, "Only 1 target permitted.")

        if self._prepared:
            raise Exception("Encoder is already prepared.")

        # ---------------------------------------
        # Initialize the base text models + tokenizer
        # ---------------------------------------

        # Setup the base model and tokenizer
        self._model = self._embeddings_model_class.from_pretrained(self.model_name).to(
            self.device
        )

        self._tokenizer = self._tokenizer_class.from_pretrained(self.model_name).to(
            self.device
        )

        # Trains to a single target
        if training_data["targets"][0]["output_type"] == COLUMN_DATA_TYPES.CATEGORICAL:
            print("CATEGORICAL")

        # --------------------------
        # Prepare the input data
        # --------------------------
        log.info("Preparing the training data")

        # Replace any empty strings with a "" placeholder and add MASK tokens.
        priming_data = add_mask(priming_data, self._tokenizer._mask_token)

        # Get the output labels in the categorical space
        labels = training_data["targets"][0]["encoded_output"].argmax(
            dim=1
        )  # Nbatch x N_classes

        N_classes = len(set(training_data["targets"][0]["unencoded_output"]))
        self._labeldict, self._tokenizer = create_label_tokens(
            N_classes,
            self._tokenizer
        )

        # Tokenize the dataset
        text = self._tokenizer(
            priming_data,
            truncation=True,
            padding=True,
            add_special_tokens=True
        )

        # Construct a dataset class and data loader
        traindata = DataLoader(
            MaskedText(text, self._tokenizer.mask_token_id, self._labeldict),
            batch_size=self._batch_size,
            shuffle=True
        )

        # --------------------------
        # PSetup the training parameters
        # -------------------------
        log.info("Training the model")
        optimizer = AdamW(self._model.parameters(), lr=self._lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            #    num_training_steps=len(dataset) * self._epochs,
        )

        # --------------------------
        # Train the model
        # -------------------------
        self._train_model(
            traindata,
            optim=optimizer,
            scheduler=scheduler,
            n_epochs=self._epochs
        )

        log.info("Text encoder is prepared!")
        self._prepared = True



    def decode(self, encoded_values_tensor, max_length=100):
        raise Exception("Decoder not implemented.")


    def _train_model(
        self,
        dataset,
        optim: AdamW,
        scheduler: get_linear_schedule_with_warmup,
        n_epochs: int,
    ):
        """
        Trains the MLM for n_epochs provided. More advanced options (i.e. early stopping should be customized.)

        Args:
        ::param dataset; dataset to train
        ::param optim; training optimizer
        ::param scheduler; learning-rate scheduler for smoother steps
        ::param n_epochs; number of epochs to train
        """
        self._model.train()

        for epoch in range(n_epochs):
            total_loss = 0

            for batch in dataset:
                optim.zero_grad()

                with LightwoodAutocast():

                    # Prepare the batch and its labels
                    inpids = batch["input_ids"].to(self.device)
                    attn = batch["attention_mask"].to(self.device)
                    labels = batch["labels"].to(self.device)

                    outputs = self._model(inpids, attention_mask=attn, labels=labels)
                    loss = outputs[0]

                total_loss += loss.item()

                # Update the weights
                loss.backward()
                optim.step()
                scheduler.step()

            self._train_callback(epoch, total_loss / len(dataset))

    def _train_callback(self, epoch, loss):
        """ Training step details """
        log.info(f"{self.name} at epoch {epoch+1} and loss {loss}!")
