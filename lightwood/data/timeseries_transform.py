from lightwood.api.types import ProblemDefinition
from lightwood.helpers.log import log
from lightwood.api import dtype
from typing import Dict
import pandas as pd

# old
import time
import copy
import dateutil
import datetime
import traceback
import numpy as np
from pathlib import Path
import multiprocessing as mp
from functools import partial
import os
import psutil
import multiprocessing as mp


def transform_timeseries(data: pd.DataFrame, dtype_dict: Dict[str, str], problem_definition: ProblemDefinition) -> pd.DataFrame:

    data, secondary_type_dict, timeseries_row_mapping, df_gb_map = _ts_reshape(data,
                                                                               dtype_dict,
                                                                               problem_definition,
                                                                               mode='predict')
    return data


def _ts_reshape(original_df, dtype_dict, problem_definition, mode='learn'):
    tss = problem_definition.timeseries_settings
    original_df = copy.deepcopy(original_df)
    gb_arr = tss.group_by if tss.group_by is not None else []
    ob_arr = tss.order_by
    window = tss.window

    if 'make_predictions' in original_df.columns:
        index = original_df[original_df['make_predictions'].map({'True': True, 'False': False, True: True, False: False}) == True]
        infer_mode = index.shape[0] == 0  # condition to trigger: make_predictions is set to False everywhere
    else:
        infer_mode = False
    original_index_list = []
    idx = 0
    for row in original_df.itertuples():
        if _make_pred(row) or infer_mode:
            original_index_list.append(idx)
            idx += 1
        else:
            original_index_list.append(None)

    original_df['original_index'] = original_index_list

    secondary_type_dict = {}
    for col in ob_arr:
        if dtype_dict[col] in (dtype.date, dtype.integer, dtype.float):
            secondary_type_dict[col] = dtype_dict[col]

    # Convert order_by columns to numbers (note, rows are references to mutable rows in `original_df`)
    for _, row in original_df.iterrows():
        for col in ob_arr:
            # @TODO: Remove if the TS encoder can handle `None`
            if row[col] is None or pd.isna(row[col]):
                row[col] = 0.0
            else:
                if dtype_dict[col] == dtype.date:
                    try:
                        row[col] = dateutil.parser.parse(
                            row[col],
                            # transaction.lmd.get('dateutil_parser_kwargs_per_column', {}).get(col, {}) # @TODO
                            **{}
                        )
                    except (TypeError, ValueError):
                        pass

                if isinstance(row[col], datetime.datetime):
                    row[col] = row[col].timestamp()

                try:
                    row[col] = float(row[col])
                except ValueError:
                    raise ValueError(f'Failed to order based on column: "{col}" due to faulty value: {row[col]}')

    if len(gb_arr) > 0:
        df_arr = []
        for _, df in original_df.groupby(gb_arr):
            df_arr.append(df.sort_values(by=ob_arr))
    else:
        df_arr = [original_df]

    last_index = original_df['original_index'].max()
    for i, subdf in enumerate(df_arr):
        if 'make_predictions' in subdf.columns and mode == 'predict':  # @TODO: make_predictions would not be a thing anymore
            if infer_mode:
                df_arr[i] = _ts_infer_next_row(subdf, ob_arr, last_index)
                last_index += 1

    if len(original_df) > 500:
        # @TODO: restore possibility to override this with args
        nr_procs = get_nr_procs(original_df)
        log.info(f'Using {nr_procs} processes to reshape.')
        pool = mp.Pool(processes=nr_procs)
        # Make type `object` so that dataframe cells can be python lists
        df_arr = pool.map(partial(_ts_to_obj, historical_columns=ob_arr + tss.historical_columns), df_arr)
        df_arr = pool.map(partial(_ts_order_col_to_cell_lists, historical_columns=ob_arr + tss.historical_columns), df_arr)
        df_arr = pool.map(partial(_ts_add_previous_rows, historical_columns=ob_arr + tss.historical_columns, window=window), df_arr)
        if tss.use_previous_target:
            df_arr = pool.map(partial(_ts_add_previous_target, predict_columns=transaction.lmd['predict_columns'], nr_predictions=nr_predictions, window=window, mode=mode), df_arr)
        pool.close()
        pool.join()
    else:
        for i in range(len(df_arr)):
            df_arr[i] = _ts_to_obj(df_arr[i], historical_columns=ob_arr + tss.historical_columns)
            df_arr[i] = _ts_order_col_to_cell_lists(df_arr[i], historical_columns=ob_arr + tss.historical_columns)
            df_arr[i] = _ts_add_previous_rows(df_arr[i], historical_columns=ob_arr + tss.historical_columns, window=window)
            if tss.use_previous_target:
                df_arr[i] = _ts_add_previous_target(df_arr[i], predict_columns=transaction.lmd['predict_columns'], nr_predictions=nr_predictions, window=window, mode=mode)

    combined_df = pd.concat(df_arr)

    if 'make_predictions' in combined_df.columns:
        combined_df = pd.DataFrame(combined_df[combined_df['make_predictions'].astype(bool) == True])
        del combined_df['make_predictions']

    if len(combined_df) == 0:
        raise Exception(f'Not enough historical context to make a timeseries prediction. Please provide a number of rows greater or equal to the window size. If you can\'t get enough rows, consider lowering your window size. If you want to force timeseries predictions lacking historical context please set the `allow_incomplete_history` advanced argument to `True`, but this might lead to subpar predictions.')

    df_gb_map = None
    if len(df_arr) > 1 and (transaction.lmd['quick_learn'] or transaction.lmd['quick_predict']):
        df_gb_list = list(combined_df.groupby(transaction.lmd['split_models_on']))
        df_gb_map = {}
        for gb, df in df_gb_list:
            df_gb_map['_' + '_'.join(gb)] = df

    timeseries_row_mapping = {}
    idx = 0

    if df_gb_map is None:
        for _, row in combined_df.iterrows():
            if not infer_mode:
                timeseries_row_mapping[idx] = int(row['original_index']) if row['original_index'] is not None and not np.isnan(row['original_index']) else None
            else:
                timeseries_row_mapping[idx] = idx
            idx += 1
    else:
        for gb in df_gb_map:
            for _, row in df_gb_map[gb].iterrows():
                if not infer_mode:
                    timeseries_row_mapping[idx] = int(row['original_index']) if row['original_index'] is not None and not np.isnan(row['original_index']) else None
                else:
                    timeseries_row_mapping[idx] = idx

                idx += 1

    del combined_df['original_index']

    return combined_df, secondary_type_dict, timeseries_row_mapping, df_gb_map

def _ts_infer_next_row(df, ob, last_index):
    last_row = df.iloc[[-1]].copy()
    if df.shape[0] > 1:
        butlast_row = df.iloc[[-2]]
        delta = (last_row[ob].values - butlast_row[ob].values).flatten()[0]
    else:
        delta = 1
    last_row.original_index = None
    last_row.index = [last_index + 1]
    last_row['make_predictions'] = True
    last_row[ob] += delta
    return df.append(last_row)


def _make_pred(row):
    return not hasattr(row, 'make_predictions') or row.make_predictions


def _ts_to_obj(df, historical_columns):
    for hist_col in historical_columns:
        df.loc[:, hist_col] = df[hist_col].astype(object)
    return df


def _ts_order_col_to_cell_lists(df, historical_columns):
    for order_col in historical_columns:
        for ii in range(len(df)):
            label = df.index.values[ii]
            df.at[label, order_col] = [df.at[label, order_col]]
    return df


def _ts_add_previous_rows(df, historical_columns, window):
    for order_col in historical_columns:
        for i in range(len(df)):
            previous_indexes = [*range(max(0, i - window), i)]

            for prev_i in reversed(previous_indexes):
                df.iloc[i][order_col].append(
                    df.iloc[prev_i][order_col][-1]
                )

            # Zero pad
            # @TODO: Remove since RNN encoder can do without (???)
            df.iloc[i][order_col].extend(
                [0] * (1 + window - len(df.iloc[i][order_col]))
            )
            df.iloc[i][order_col].reverse()
    return df


def _ts_add_previous_target(df, predict_columns, nr_predictions, window, mode):
    for target_column in predict_columns:
        previous_target_values = list(df[target_column])
        del previous_target_values[-1]
        previous_target_values = [None] + previous_target_values

        previous_target_values_arr = []
        for i in range(len(previous_target_values)):
            prev_vals = previous_target_values[max(i - window, 0):i + 1]
            arr = [None] * (window - len(prev_vals) + 1)
            arr.extend(prev_vals)
            previous_target_values_arr.append(arr)

        df[f'__mdb_ts_previous_{target_column}'] = previous_target_values_arr
        for timestep_index in range(1, nr_predictions):
            next_target_value_arr = list(df[target_column])
            for del_index in range(0, min(timestep_index, len(next_target_value_arr))):
                del next_target_value_arr[0]
                next_target_value_arr.append(None)
            df[f'{target_column}_timestep_{timestep_index}'] = next_target_value_arr

    # drop rows with incomplete target info.
    if mode == 'learn':
        for target_column in predict_columns:
            for col in [f'{target_column}_timestep_{i}' for i in range(1, nr_predictions)]:
                if 'make_predictions' not in df.columns:
                    df['make_predictions'] = True
                df.loc[df[col].isna(), ['make_predictions']] = False

    return df


def get_nr_procs(df=None, max_processes=None, max_per_proc_usage=None):
    if os.name == 'nt':
        return 1
    else:
        available_mem = psutil.virtual_memory().available
        if max_per_proc_usage is None or type(max_per_proc_usage) not in (int, float):
            try:
                import mindsdb_worker
                import ray
                max_per_proc_usage = 0.2 * pow(10,9)
            except:
                max_per_proc_usage = 3 * pow(10, 9)
            if df is not None:
                max_per_proc_usage += df.memory_usage(index=True, deep=True).sum()
        proc_count = int(min(mp.cpu_count(), available_mem // max_per_proc_usage)) - 1
        if isinstance(max_processes, int):
            proc_count = min(proc_count, max_processes)
        return max(proc_count, 1)