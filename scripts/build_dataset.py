'''
build train, test, val sets with clean utterance for whisper fine-tuning
'''


from datetime import datetime
import warnings
import os
import re
import json
import configparser
from glob import glob
import pandas as pd
from pydub import AudioSegment
from datasets import load_dataset
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')


def words_to_numbers(input_text):
    """
    inversely convert words to number

    :param input_text: the input uttrance
    :type input_text: str
    :return: the input utterance with converted numbers
    :rtype: str
    """
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


def split_data(full_df):
    """
    split CORAAL ver. 2021 into training, test, val sets, stratified by location

    :return: _description_
    :rtype: _type_
    """
    full_df['geo'] = full_df['Spkr'].str[:3]
    train_set, temp_set = train_test_split(
        full_df, test_size=0.5, random_state=42, stratify=full_df['geo'])
    test_set, val_set = train_test_split(
        temp_set, test_size=0.6, random_state=42, stratify=temp_set['geo'])
    for geo_subset in [train_set, test_set, val_set]:
        geo_subset = geo_subset.groupby('Spkr', group_keys=False).apply(
            lambda x: x.sample(min(len(x), 1), random_state=42)
        )
    print(train_set.shape)
    print(test_set.shape)
    print(val_set.shape)
    return train_set, test_set, val_set


def clean_txt(txtin):
    """
    text preprocessing for clean utterance

    :param txtin: the utterance
    :type txtin: str
    :return: the cleaned utterance
    :rtype: str
    """
    pattern = re.compile(r'/.*?/|\[.*?\]|<.*?>|\(.*?\)')
    if pattern.search(txtin):
        return ""
    else:
        txtout = words_to_numbers(txtin)
        txtout = re.sub(r'[^a-zA-Z0-9\s\']', "", txtout)
        return txtout.lower().strip()


def load_trans():
    """
    load coraal transcripts as a dataframe

    :param file_path: the path to all transcripts
    :type file_path: str
    """
    total_df = pd.DataFrame()
    all_files = glob("../coraal-files/*.json")
    for each_file in all_files:
        with open(each_file, "r") as json_f:
            sub = json.load(json_f)
        for _, files in sub.items():
            par_txt_df = pd.read_csv(files['txt'], sep="\t")
            par_txt_df = par_txt_df[~par_txt_df['Spkr'].str.contains(r'(int|misc)', case=False)]
            par_txt_df['clean_tran'] = par_txt_df["Content"].apply(clean_txt)
            par_txt_df.replace('', pd.NA, inplace=True)
            par_txt_df.dropna(inplace=True)
            par_txt_df['loc'] = [files['wav']]*len(par_txt_df)
            total_df = pd.concat([total_df, par_txt_df], ignore_index=True)
    total_df.to_csv("../total_df.tsv", sep="\t", index=False)
    print(f"Number of utterance: {len(total_df)}")


def resample_and_slice_audio(sub_df, out_dir, target_sample_rate=16000):
    """
    resample and slice the audio file into utterance level audio slides

    :param sub_df: the train/test split
    :type sub_df: pd.DataFrame
    :param out_dir: the output directory to save all audio clips
    :type out_dir: str
    :param target_sample_rate: the resample rate, defaults to 16000
    :type target_sample_rate: int, optional
    """
    meta = pd.DataFrame()
    for _, speaker_group in sub_df.groupby('Spkr'):
        audio_file = speaker_group['loc'].iloc[0]
        audio = AudioSegment.from_file(audio_file).set_frame_rate(target_sample_rate)
        file_name = os.path.splitext(os.path.basename(audio_file))[0]
        cut_loc = []
        for i, (start_time, end_time) in speaker_group[['StTime', 'EnTime']].iterrows():
            new_file_name = f"{file_name}_{i}.wav"
            cut_loc.append(new_file_name)
            sliced_audio = audio[start_time * 1000:end_time * 1000]
            sliced_audio.export(os.path.join(out_dir, new_file_name), format='wav')
        speaker_group['cut_loc'] = cut_loc
        speaker_group = speaker_group[['cut_loc', 'clean_tran']]
        meta = pd.concat([meta, speaker_group], ignore_index=True)
    meta.columns = ['file_name', 'transcription']
    meta.to_csv(os.path.join(out_dir, "metadata.csv"), index=False)


def resample_and_slice_audio_val(sub_df, out_dir, target_sample_rate=16000):
    """
    resample and slice the audio recordings for validation split

    :param sub_df: the validation split
    :type sub_df: pd.DataFrame
    :param out_dir: the output direcotry to save all audio clips
    :type out_dir: str
    :param target_sample_rate: the resample rate, defaults to 16000
    :type target_sample_rate: int, optional
    """
    for geo, geo_df in sub_df.groupby('geo'):
        geo_out_dir = os.path.join(out_dir, geo)
        os.makedirs(geo_out_dir, exist_ok=True)
        meta = pd.DataFrame()
        for _, speaker_group in geo_df.groupby('Spkr'):
            audio_file = speaker_group['loc'].iloc[0]
            audio = AudioSegment.from_file(audio_file).set_frame_rate(target_sample_rate)
            file_name = os.path.splitext(os.path.basename(audio_file))[0]
            cut_loc = []
            for i, (start_time, end_time) in speaker_group[['StTime', 'EnTime']].iterrows():
                new_file_name = f"{file_name}_{i}.wav"
                cut_loc.append(new_file_name)
                sliced_audio = audio[start_time * 1000:end_time * 1000]
                sliced_audio.export(os.path.join(geo_out_dir, new_file_name), format='wav')
            speaker_group['cut_loc'] = cut_loc
            speaker_group = speaker_group[['cut_loc', 'clean_tran']]
            meta = pd.concat([meta, speaker_group], ignore_index=True)
        meta.columns = ['file_name', 'transcription']
        meta.to_csv(os.path.join(geo_out_dir, "metadata.csv"), index=False)


if __name__ == "__main__":
    begin_time = datetime.now()
    config = configparser.ConfigParser()
    config.read("config.ini")
    if os.path.exists("../total_df.tsv"):
        total_df = pd.read_csv("../total_df.tsv", sep="\t")
    else:
        load_trans()
        total_df = pd.read_csv("../total_df.tsv", sep="\t")
    os.makedirs(config['DATA']['data'], exist_ok=True)
    os.makedirs(config['DATA']['train'], exist_ok=True)
    os.makedirs(config['DATA']['test'], exist_ok=True)
    os.makedirs(config['DATA']['val'], exist_ok=True)
    train_df, test_df, val_df = split_data(total_df)
    print("Resample and slice the training split")
    resample_and_slice_audio(train_df, config['DATA']['train'])
    print("Resample and slice the test split")
    resample_and_slice_audio(test_df, config['DATA']['test'])
    print("Resample and slice the validation split")
    resample_and_slice_audio_val(val_df, config['DATA']['val'])
    print(f"Total running time: {datetime.now() - begin_time}")