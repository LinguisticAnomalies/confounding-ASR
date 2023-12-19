'''
Get descriptive stats for CORAAL
'''
from glob import glob
import os
import re
import configparser
import pandas as pd


def remove_suffix(filename):
    pattern = re.compile(r'_\d+\.wav$')
    return re.sub(pattern, '', filename)


config = configparser.ConfigParser()
config.read("config.ini")
total_df = pd.read_csv("../total_df.csv")
grouped_df = total_df.groupby('Spkr')['clean_tran'].agg(lambda x: ' '.join(x)).reset_index()
grouped_df['geo'] = grouped_df['Spkr'].str[:3]
# ------ train split -------
print("---- train split -----")
train_meta = pd.read_csv(
    os.path.join(config['DATA']['train'], 'metadata.csv')
)
train_meta['geo'] = train_meta['file_name'].str[:3]
print("By utterance")
print(train_meta['geo'].value_counts())
train_meta['clean_filename'] = train_meta['file_name'].apply(remove_suffix)
grouped_train = train_meta.groupby('clean_filename')['transcription'].agg(lambda x: ' '.join(x)).reset_index()
grouped_train['geo'] = grouped_train['clean_filename'].str[:3]
print("By speaker")
print(grouped_train['geo'].value_counts())
# ------ test split -------
print("---- test split -----")
test_meta = pd.read_csv(
    os.path.join(config['DATA']['test'], 'metadata.csv')
)
test_meta['geo'] = test_meta['file_name'].str[:3]
print("By utterance")
print(test_meta['geo'].value_counts())
test_meta['clean_filename'] = test_meta['file_name'].apply(remove_suffix)
grouped_test = test_meta.groupby('clean_filename')['transcription'].agg(lambda x: ' '.join(x)).reset_index()
grouped_test['geo'] = grouped_test['clean_filename'].str[:3]
print("By speaker")
print(grouped_test['geo'].value_counts())
# ------ vali split -------
print("---- validation split -----")
vali_meta = pd.DataFrame()
for loc in ("DCA", "DCB", "PRV", "ROC", "ATL"):
    sub_meta = pd.read_csv(
        os.path.join(f"{config['DATA']['vali']}{loc}", "metadata.csv")
    )
    vali_meta = pd.concat([vali_meta, sub_meta])
vali_meta['geo'] = vali_meta['file_name'].str[:3]
print("By utterance")
print(vali_meta['geo'].value_counts())
vali_meta['clean_filename'] = vali_meta['file_name'].apply(remove_suffix)
grouped_vali = vali_meta.groupby('clean_filename')['transcription'].agg(lambda x: ' '.join(x)).reset_index()
grouped_vali['geo'] = grouped_vali['clean_filename'].str[:3]
print("By speaker")
print(grouped_vali['geo'].value_counts())