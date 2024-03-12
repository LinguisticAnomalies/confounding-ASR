from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from glob import glob
import tarfile
import os
import re
import configparser
import numpy as np
import pandas as pd
from datasets import load_dataset

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

def clean_check(txtin):
    """
    preprocessing the transcript

    :param txtin: the utterance-level transcript
    :type txtin: str
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
    all_files = glob(f"{file_path}/ATL*.txt")
    all_audio_files = glob(f"{file_path}/*.wav")
    for file in all_files:
        file_name = file.split("/")[-1].split(".")[0]
        audio_loc = [element for element in all_audio_files if file_name in element]
        tran_df = pd.read_csv(file, sep="\t")
        # add audio location
        tran_df['loc'] = audio_loc*len(tran_df)
        # drop interviewer and misc rows
        tran_df = tran_df[~tran_df['Spkr'].str.contains(
            r'(int|misc)', case=False)]
        total_df = pd.concat([total_df, tran_df], ignore_index=True)
    print(f"Number of utterance BEFORE cleaning: {len(total_df)}")
    total_df['clean_tran'] = total_df["Content"].apply(clean_check)
    total_df['clean_tran'] = total_df['clean_tran'].str.lower()
    total_df['clean_tran'] = total_df['clean_tran'].str.strip()
    total_df['clean_tran'].replace('', np.nan, inplace=True)
    total_df.dropna(inplace=True)
    print(f"Number of utterance AFTER cleaning: {len(total_df)}")
    return total_df

def list_files_in_tar_gz(folder_name):
    with tarfile.open(folder_name, 'r:gz') as tar:
        file_list = [member.name for member in tar.getmembers() if member.isfile()]
    return file_list


if __name__ == "__main__":
    start_time = datetime.now()
    config = configparser.ConfigParser()
    config.read("config.ini")
    trans = glob("../coraal-files/*.tsv")
