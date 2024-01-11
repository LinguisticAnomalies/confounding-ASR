'''
A not-so-smart way to create audiofolder dataset for CORAAL
'''


from datetime import datetime
import warnings
import os
import re
import configparser
from glob import glob
import pandas as pd
import numpy as np
from pydub import AudioSegment
from datasets import load_dataset
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')

def words_to_numbers(input_text):
    # Dictionary mapping words to numbers
    word_to_number = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10', 'eleven': '11', 'twelve': '12', 'thirteen': '13',
        'fourteen': '14', 'fifteen': '15', 'sixteen': '16', 'seventeen': '17',
        'eighteen': '18', 'nineteen': '19', 'twenty': '20', 'thirty': '30',
        'forty': '40', 'fifty': '50', 'sixty': '60', 'seventy': '70',
        'eighty': '80', 'ninety': '90'
    }
    # Pattern to match words representing numbers
    pattern = re.compile(r'\b(?:' + '|'.join(word_to_number.keys()) + r')\b', re.IGNORECASE)
    # Replace words with corresponding numbers
    result = pattern.sub(lambda match: word_to_number[match.group(0).lower()], input_text)
    pattern = re.compile(r'(\b\d{1,2})-(\d{2}\b)')
    # 20-16 -> 2016, 5-30 -> 530
    result = pattern.sub(lambda match: match.group(1) + match.group(2), result)
    pattern_general = re.compile(r'(\d+)-(\d+)')
    result = pattern_general.sub(lambda match: str(int(match.group(1)) + int(match.group(2))), result)
    return result


def clean_text(txtin):
    """
    preprocessing the transcript

    :param txtin: the utterance-level transcript
    :type txtin: str
    :return: the utterance after pre-processing
    :rtype: str
    """
    pattern = re.compile(r'/.*?/|\[.*?\]|<.*?>|\(.*?\)')
    if pattern.search(txtin):
        return ""
    else:
        # words to number
        txtout = words_to_numbers(txtin)
        # filling words
        txtout = re.sub(r'(\s+|^)uh(\s+|$)', ' ah ', txtout)
        txtout = re.sub(r'(\s+|^)uhm(\s+|$)', ' um ', txtout)
        txtout = re.sub(r'\s+', ' ', txtout)
        # restarts
        txtout = re.sub(r'-(?!\w)', "", txtout)
        # punctuation
        txtout = re.sub(r'[^a-zA-Z0-9\s\']', "", txtout)
        return txtout


def load_trans(file_path):
    """
    load coraal transcripts as a dataframe

    :param file_path: the path to all transcripts
    :type file_path: str
    """
    total_df = pd.DataFrame()
    all_files = glob(f"{file_path}/*.txt")
    all_audio_files = glob(f"{file_path}/*.wav")
    total_df_list = []
    for file in all_files:
        file_name = file.split("/")[-1].split(".")[0]
        audio_loc = [element for element in all_audio_files if file_name in element]
        trans_df = pd.read_csv(file, sep="\t")
        trans_df['loc'] = audio_loc * len(trans_df)
        trans_df = trans_df[~trans_df['Spkr'].str.contains(r'(int|misc)', case=False)]
        trans_df['clean_tran'] = trans_df["Content"].apply(clean_text).str.lower().str.strip()
        trans_df['clean_tran'].replace('', np.nan, inplace=True)
        trans_df.dropna(inplace=True)
        total_df_list.append(trans_df)
    total_df = pd.concat(total_df_list, ignore_index=True)
    total_df.dropna(subset=['clean_tran'], inplace=True)
    print(f"Number of utterance AFTER cleaning: {len(total_df)}")
    return total_df


def resample_and_slice_audio(
        speaker_group, out_dir, target_sample_rate=16000, vali_set=False):
    """
    resample, slice, and save the audio file

    :param speaker_group: a DataFrame containing audio and metadata for a single speaker.
    :type speaker_group: pd.DataFrame
    :param out_dir: the output folder for train/test split
    :type out_dir: str
    :param target_sample_rate: the target sample rate for resampling, defaults to 16000
    :type target_sample_rate: int, optional
    :param vali_set: if the dataset is validation set, defaults to false
    :type target_sample_rate: bool, optional
    """
    audio_file = speaker_group['loc'].iloc[0]
    # load audio
    audio = AudioSegment.from_file(audio_file)
    # resample
    audio = audio.set_frame_rate(target_sample_rate)
    file_name = os.path.splitext(os.path.basename(audio_file))[0]
    start_times = (speaker_group['StTime'] * 1000).values
    end_times = (speaker_group['EnTime'] * 1000).values
    cut_loc = []
    if vali_set:
        # slice audio file base on location
        geo = speaker_group['geo'].iloc[0]
        out_dir = os.path.join(out_dir, geo)
        os.makedirs(out_dir, exist_ok=True)
    for i in range(len(speaker_group)):
        start_time = start_times[i]
        end_time = end_times[i]
        new_file_name = f"{file_name}_{i}.wav"
        cut_loc.append(new_file_name)
        sliced_audio = audio[start_time:end_time]
        if not os.path.exists(os.path.join(out_dir, new_file_name)):
            sliced_audio.export(os.path.join(out_dir, new_file_name), format='wav')
        else:
            continue
    speaker_group['cut_loc'] = cut_loc
    speaker_group = speaker_group[['cut_loc', 'clean_tran']]
    speaker_group.columns = ['file_name', 'transcription']
    # save the metadata file
    meta_file = os.path.join(out_dir, 'metadata.csv')
    # Check if the file exists, and if not, write the header
    if not os.path.exists(meta_file):
        speaker_group.to_csv(meta_file, index=False)
    else:
        speaker_group.to_csv(meta_file, mode="a", header=False, index=False)


def build_dataset(total_df, data_dir):
    """
    Build a dataset from preprocessed audio slices and metadata.
    only use 50% of the CORAAL

    :param total_df: The dataframe containing utterance-level data.
    :param data_dir: The base directory where audio files and metadata will be stored.
    """
    os.makedirs(data_dir, exist_ok=True)
    # 50%, 20%, 30% training, test, validation set
    use_size = 0.3
    test_size = 0.2
    total_df["geo"] = total_df['Spkr'].str[:3]
    # make sure that speaker's full speeach remains in one of the two sets
    grouped = total_df.groupby(['Spkr', 'geo'])['clean_tran'].apply(" ".join).reset_index()
    # training and validation
    train_grouped, vali_grouped = train_test_split(grouped, test_size=use_size, random_state=42)
    # training to training and test
    train_grouped, test_grouped = train_test_split(train_grouped, test_size=test_size, random_state=42)
    # training, test, validation set
    train_df = total_df[total_df['Spkr'].isin(train_grouped['Spkr']) & (total_df['geo'].isin(train_grouped['geo']))]
    vali_df = total_df[total_df['Spkr'].isin(vali_grouped['Spkr']) & (total_df['geo'].isin(vali_grouped['geo']))]
    test_df = total_df[total_df['Spkr'].isin(test_grouped['Spkr']) & (total_df['geo'].isin(test_grouped['geo']))]
    # Create metadata dataframes
    train_metadata = train_df[['loc', 'clean_tran']]
    test_metadata = test_df[['loc', 'clean_tran']]
    vali_metadata = vali_df[['loc', 'clean_tran']]
    train_metadata.columns = \
        test_metadata.columns = \
            vali_metadata.columns = \
                ['file_name', 'transcription']
    # Create train/test folders
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    vali_dir = os.path.join(data_dir, "vali")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(vali_dir, exist_ok=True)
    # Resample and slice the audio
    print("Resampling and slicing the training and test set...")
    for df, dir in [(train_df, train_dir), (test_df, test_dir)]:
        for _, group in df.groupby('Spkr'):
            resample_and_slice_audio(group, dir)
    # Resample and slice the audio for the validation set
    print("Resampling and slicing the validation set...")
    for _, group in vali_df.groupby('Spkr'):
        resample_and_slice_audio(group, vali_dir, vali_set=True)


if __name__ == "__main__":
    begin_time = datetime.now()
    config = configparser.ConfigParser()
    config.read("config.ini")
    tran_df = load_trans(config['DATA']['data'])
    tran_df.to_csv("../total_df.csv", index=False)
    build_dataset(tran_df, config['DATA']['dataset'])
    print(f"Total running time: {datetime.now() - begin_time}")
