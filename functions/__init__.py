import pandas as pd
import numpy as np
import os

from tensorflow.keras.preprocessing.sequence import pad_sequences


def get_all_data_for_train():
    all_data = pd.DataFrame()
    for fname in os.listdir(r'.\data'):
        df = get_data_for_train(r'.\data\{fname}'.format(fname=fname))
        all_data = pd.concat([all_data, df], ignore_index=True)
    return all_data


def get_data_for_train(filename):
    good_point = pd.read_excel(filename, sheet_name=0)
    abnormality = pd.read_excel(filename, sheet_name=1)
    good_point['target'] = 'good point'
    abnormality['target'] = 'abnormality'
    df = pd.concat([good_point, abnormality], ignore_index=True)
    return df


def get_data_for_test(filename):
    df = pd.read_excel(filename)
    return df


def get_sequences(df):
    stages_and_temps = []
    stages_and_temps_seq = []

    for index, item in df.iterrows():
        if pd.notna(item['Sample']):
            if stages_and_temps_seq:
                stages_and_temps.append(stages_and_temps_seq)
                stages_and_temps_seq = []

        stages_and_temps_seq.append([item['Stage'], item['Temp']])
        if index == len(df) - 1:
            stages_and_temps.append(stages_and_temps_seq)

    X_padded = pad_sequences(stages_and_temps, dtype='float32', maxlen=23)
    return stages_and_temps, X_padded


def get_sequences_and_targets(df):
    stages_and_temps = []
    stages_and_temps_seq = []
    y = []

    for index, item in df.iterrows():
        if pd.notna(item['Sample']):
            if stages_and_temps_seq:
                stages_and_temps.append(stages_and_temps_seq)
                y.append(0 if df.loc[index - 1]['target'] == 'good point' else 1)
                stages_and_temps_seq = []

        stages_and_temps_seq.append([item['Stage'], item['Temp']])
        if index == len(df) - 1:
            stages_and_temps.append(stages_and_temps_seq)
            y.append(0 if df.loc[index - 1]['target'] == 'good point' else 1)

    X = pad_sequences(stages_and_temps, dtype='float32')
    y = np.array(y)
    return X, y
