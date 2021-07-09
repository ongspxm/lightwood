import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from pyts.image import GramianAngularField

from lightwood.helpers.device import get_devices
from lightwood.helpers.torch import LightwoodAutocast
from lightwood.encoders.encoder_base import BaseEncoder


class GramianTSEncoder(BaseEncoder):
    def __init__(self, is_target=False, img_size=10):  # 12 => img_size should be (window+1)
        super().__init__(is_target)
        self.device, _ = get_devices()
        self.img_size = img_size
        self.hidden_dimension = 128
        self.epochs = 200

        self.SumEnc = GramianAngularField(image_size=self.img_size, method='summation')
        self.DiffEnc = GramianAngularField(image_size=self.img_size, method='difference')

        self.cnn_enc = nn.Sequential(
            nn.Linear(2*(img_size**2), 2*img_size), nn.ReLU(True),
            nn.Linear(2*img_size, img_size), nn.ReLU(True)).to(self.device)

        self.cnn_dec = nn.Sequential(
            nn.Linear(img_size, 2*img_size), nn.ReLU(True),
            nn.Linear(2*img_size, 2*(img_size**2)), nn.Tanh()).to(self.device)

    def prepare(self, priming_data):
        if self._prepared:
            raise Exception('You can only call "prepare" once for a given encoder.')

        if isinstance(priming_data, list):
            priming_data = np.array(priming_data, dtype=np.float)
            priming_data = np.nan_to_num(priming_data)

        batch_size = priming_data.shape[0]
        summed = self.SumEnc.fit_transform(priming_data).reshape(batch_size, -1)
        diffed = self.DiffEnc.fit_transform(priming_data).reshape(batch_size, -1)
        concatenated = np.concatenate([summed, diffed], axis=-1)
        criterion = torch.nn.MSELoss()
        optimizer = Adam(params=list(self.cnn_enc.parameters())+list(self.cnn_dec.parameters()))

        with LightwoodAutocast():
            for epoch in range(self.epochs):
                running_loss = 0.0
                for i in range(0, concatenated.shape[0], batch_size):
                    data = concatenated[i:i+batch_size, :]
                    inputs = torch.Tensor(data).to(self.device)
                    labels = torch.Tensor(data).to(self.device)
                    optimizer.zero_grad()
                    outputs = self.cnn_dec(self.cnn_enc(inputs)).to(self.device)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, loss))
                    running_loss = 0.0
        self._prepared = True

    def encode(self, column_data):
        """
        :param column_data: numpy array with (N_samples, TS_length) shape
        :return: (N_samples, encoded_TS_length) matrix
        """
        if not self._prepared:
            raise Exception('You need to call "prepare" before calling "encode" or "decode".')

        if isinstance(column_data, list):
            column_data = np.array(column_data, dtype=np.float)
            column_data = np.nan_to_num(column_data)

        batch_size = column_data.shape[0]
        summed = self.SumEnc.transform(column_data).reshape(batch_size, -1)
        diffed = self.DiffEnc.transform(column_data).reshape(batch_size, -1)
        concatenated = torch.Tensor(np.concatenate([summed, diffed], axis=-1)).to(self.device)
        encoded = self.cnn_enc(concatenated).to(self.device)

        return encoded

    def decode(self, data):
        raise NotImplementedError()




if __name__ == '__main__':
    import pandas as pd
    from pyts.datasets import load_gunpoint
    from sklearn.metrics import balanced_accuracy_score
    import mindsdb_native

    # X.shape = (50, 150)
    X, Xt, Y, Yt = load_gunpoint(return_X_y=True)

    encoder = GramianTSEncoder()
    encoder.prepare(X)

    enc = encoder.encode(X).cpu().detach().numpy()

    df = pd.DataFrame(enc, columns=[f'a_{i}' for i in range(enc.shape[1])])
    df['target'] = Y

    p = mindsdb_native.Predictor(name='gramian')
    p.learn(from_data=df, to_predict='target')

    tenc = encoder.encode(Xt).cpu().detach().numpy()
    test_df = pd.DataFrame(tenc, columns=[f'a_{i}' for i in range(tenc.shape[1])])
    test_df['target'] = Yt

    result = p.test(when_data=test_df, accuracy_score_functions=balanced_accuracy_score)
    print(result)