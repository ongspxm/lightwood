import math
import torch
import numpy as np
from lightwood.encoder.numeric import NumericEncoder
from lightwood.helpers.log import log


class TsNumericEncoder(NumericEncoder):
    """
    Variant of vanilla numerical encoder, supports dynamic mean re-scaling
    """
    is_timeseries_encoder: bool = True

    def __init__(self, is_target: bool = False, positive_domain: bool = False, grouped_by=None):
        super(TsNumericEncoder, self).__init__(is_target=is_target, positive_domain=positive_domain)
        # time series normalization params
        self.normalizers = None
        self.group_combinations = None
        self.dependencies = grouped_by
        self.output_size = 2 if is_target else 3

    def encode(self, data, dependency_data={}):
        """
        :param dependency_data: dict with grouped_by column info, to retrieve the correct normalizer for each datum
        """  # noqa
        if not self.is_prepared:
            raise Exception('You need to call "prepare" before calling "encode" or "decode".')
        if not dependency_data:
            dependency_data = {'__default': [None] * len(data)}

        ret = []
        for real, group in zip(data, list(zip(*dependency_data.values()))):
            try:
                real = float(real)
            except Exception:
                try:
                    real = float(real.replace(',', '.'))
                except Exception:
                    real = None
            if self.is_target:
                vector = [0] * 2
                if group is not None and self.normalizers is not None:
                    try:
                        mean = self.normalizers[frozenset(group)].abs_mean
                    except KeyError:
                        # novel group-by, we use default normalizer mean
                        mean = self.normalizers['__default'].abs_mean
                else:
                    mean = self._abs_mean

                if real is not None:
                    vector[0] = 1 if real < 0 and not self.positive_domain else 0
                    vector[1] = real / mean if mean != 0 else real
                else:
                    raise Exception(f'Can\'t encode target value: {real}')

            else:
                vector = [0] * 3
                try:
                    if real is not None:
                        vector[0] = 1
                        vector[1] = 1 if real < 0 and not self.positive_domain else 0
                        vector[2] = real / self._abs_mean
                except Exception as e:
                    log.error(f'Can\'t encode input value: {real}, exception: {e}')

            ret.append(vector)

        return torch.Tensor(ret)

    def decode(self, encoded_values, decode_log=None, dependency_data=None):
        if not self.is_prepared:
            raise Exception('You need to call "prepare" before calling "encode" or "decode".')

        if decode_log is None:
            decode_log = self.decode_log

        ret = []
        if not dependency_data:
            dependency_data = {'__default': [None] * len(encoded_values)}
        if isinstance(encoded_values, torch.Tensor):
            encoded_values = encoded_values.tolist()

        for vector, group in zip(encoded_values, list(zip(*dependency_data.values()))):
            if self.is_target:
                if np.isnan(vector[0]) or vector[0] == float('inf') or np.isnan(vector[1]) or vector[1] == float('inf'):
                    log.error(f'Got weird target value to decode: {vector}')
                    real_value = pow(10, 63)
                else:
                    if decode_log:
                        sign = -1 if vector[0] > 0.5 else 1
                        try:
                            real_value = math.exp(vector[1]) * sign
                        except OverflowError:
                            real_value = pow(10, 63) * sign
                    else:
                        if group is not None and self.normalizers is not None:
                            try:
                                mean = self.normalizers[frozenset(group)].abs_mean
                            except KeyError:
                                # decode new group with default normalizer
                                mean = self.normalizers['__default'].abs_mean
                        else:
                            mean = self._abs_mean

                        real_value = vector[1] * mean if mean != 0 else vector[1]

                    if self.positive_domain:
                        real_value = abs(real_value)

                    if self._type == 'int':
                        real_value = int(round(real_value, 0))

            else:
                if vector[0] < 0.5:
                    ret.append(None)
                    continue

                real_value = vector[2] * self._abs_mean

                if self._type == 'int':
                    real_value = round(real_value)

            ret.append(real_value)
        return ret
