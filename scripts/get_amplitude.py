import os
import re
from functools import lru_cache
import configparser
import argparse
import shutil
from typing import List, Tuple, Optional
import time
from dataclasses import dataclass
from glob import glob
from tqdm import tqdm
import librosa
from pydub import AudioSegment
import numpy as np
import pandas as pd
import seaborn as sns
import textgrid
import matplotlib.pyplot as plt


@dataclass
class SpeakingInterval:
    start_time: float
    end_time: float
    type: str
    text: Optional[str] = None  # can be None for silence intervals

def parge_args():
    """
    add argument parser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cut_audio", action="store_true",
        help="""if cut the audio first"""
    )
    return parser.parse_args()


def get_amplitudes(input_path, audio_type, expected_sr=44100):
    """
    get mean amplitude of each audio clip given a folder

    Args:
        input_path (str): the folder containing the audio clips for calculating mean amplitude
        expected_sr (int, optional): sampling rate. Defaults to 44100.

    Returns:
        list: a list of dictionary
    """
    mean_amplitudes = []
    component = input_path.split("/")[-1]
    audio_files = glob(f"{input_path}/*.wav")
    for audio_file in tqdm(
        audio_files, total=len(audio_files), desc=f"Processing {audio_type} audio clips"):
        filename = os.path.basename(audio_file)
        base_filename = os.path.splitext(filename)[0]
        y, _ = librosa.load(path=audio_file, sr=expected_sr)
        print(y)
        print("----------")
        db = 20*np.log10(np.abs(y) + 1e-5)
        mean_amplitudes.append(
            {"subset": component, 
             "filename": base_filename,
             "type": audio_type, 
             "amplitudes": np.mean(db)})
        time.sleep(1)
    return mean_amplitudes


def plot_separate_violins(df):
    """violin plot for each subset in CORAAL


    Args:
        df (pandas.DataFrame): dataframe containing amplitidues for each clip
    """
    # fill silence na with 0
    df.fillna(0, inplace=True)
    print(df.shape)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), height_ratios=[1, 1, 1])
    silence_stats = df[df['type'] == 'silence'].groupby('subset')['amplitudes'].agg(['mean', 'std', 'max'])
    speech_stats = df[df['type'] == 'speech'].groupby('subset')['amplitudes'].agg(['mean', 'std', 'max'])
    # average SNR per location
    pivot_df = df.pivot_table(
        index=['subset', 'filename'],
        columns='type',
        values='amplitudes',
        aggfunc='first'
    ).reset_index()
    pivot_df['snr'] = pivot_df['speech'] - pivot_df['silence']
    snr_by_location = pivot_df.groupby('subset')['snr'].agg(['mean', 'std']).reset_index()
    # upper plot
    silence_data = df[df['type'] == 'silence']
    sns.violinplot(data=silence_data, 
                  x='subset', 
                  y='amplitudes',
                  ax=ax1,
                  color='lightblue',
                  inner='quartile')
    ax1.set_title('Silence Amplitudes by Location, Silence Audio Clips')
    ax1.set_xlabel('')  # Remove x label from upper plot
    ax1.set_ylabel('Amplitude (dB)')
    # add mean
    offset = (ax1.get_ylim()[1] - ax1.get_ylim()[0]) * 0.02
    for idx, subset in enumerate(silence_stats.index):
        stats = silence_stats.loc[subset]
        ax1.text(idx, stats['max'] + offset, 
                f'mean={stats["mean"]:.1f}\nstd={stats["std"]:.1f}', 
                horizontalalignment='center',
                verticalalignment='bottom',
                color='red',
                fontweight='bold')
    # middle plot
    speech_data = df[df['type'] == 'speech']
    sns.violinplot(data=speech_data, 
                  x='subset', 
                  y='amplitudes',
                  ax=ax2,
                  color='lightgreen',
                  inner='quartile')
    ax2.set_title('Speech Amplitudes by Location, Speech Audio Clips')
    ax2.set_xlabel('Location')
    ax2.set_ylabel('Amplitude (dB)')
    # add mean
    offset = (ax2.get_ylim()[1] - ax2.get_ylim()[0]) * 0.02
    for idx, subset in enumerate(speech_stats.index):
        stats = speech_stats.loc[subset]
        ax2.text(idx, stats['max'] + offset, 
                f'mean={stats["mean"]:.1f}\nstd={stats["std"]:.1f}', 
                horizontalalignment='center',
                verticalalignment='bottom',
                color='red',
                fontweight='bold')
    # lower plot    
    sns.violinplot(data=pivot_df, 
                  x='subset', 
                  y='snr',
                  ax=ax3,
                  color='lightgray',
                  inner='quartile')
    ax3.set_title('Signal-to-Noise Ratio (SNR) Distribution by Location')
    ax3.set_xlabel('Location')
    ax3.set_ylabel('SNR (dB)')
    max_snr = pivot_df.groupby('subset')['snr'].max()
    offset = (ax3.get_ylim()[1] - ax3.get_ylim()[0]) * 0.02
    for idx, row in snr_by_location.iterrows():
        ax3.text(idx, max_snr[row['subset']] + offset, 
                f'mean={row["mean"]:.1f}\nstd={row["std"]:.1f}', 
                horizontalalignment='center',
                verticalalignment='bottom',
                color='red',
                fontweight='bold')
    # rotation
    ax1.tick_params(axis='x', rotation=45)
    ax2.tick_params(axis='x', rotation=45)
    ax3.tick_params(axis='x', rotation=45)
    # title and save to file
    fig.suptitle('Distribution of Amplitudes by Location and Type', y=1.02)
    plt.tight_layout()
    plt.savefig("../amplitudes_violin_plot.png", 
                dpi=300,
                bbox_inches='tight', 
                pad_inches=0.1,
                format='png')


@lru_cache(maxsize=128)
def load_tiers(file_path):
    """
    load and merge the specified tiers from a TextGrid file, sorted by start time

    Args:
        file_path (str): path to the TextGrid file

    Returns:
        list: lists of SpeakingInterval objects sorted by start time, including silence intervals
    """
    tg = textgrid.TextGrid.fromFile(file_path)
    all_intervals = []
    
    for tier in tg.tiers:
        intervals = [
            SpeakingInterval(
                start_time=interval.minTime,
                end_time=interval.maxTime,
                type="speech",
                text=text if (text := interval.mark.strip()) else None
            )
            for interval in tier
            if interval.mark.strip()
        ]
        all_intervals.extend(intervals)
    
    return sorted(all_intervals, key=lambda x: x.start_time)


def find_silence_gaps(intervals, min_gap=0.05):
    """
    find silence gaps between utterances

    Args:
        intervals (list): a list of tiers from TextGrid 
        min_gap (float, optional): mininal gap for silence. Defaults to 0.05.

    Returns:
        list: a list of SpeakingInterval for silences
    """
    if not intervals:
        return []
    current_end = intervals[0].end_time
    return [
        SpeakingInterval(
            start_time=current_end,
            end_time=intervals[i].start_time,
            type="silence",
            text=None
        )
        for i in range(1, len(intervals))
        if (gap_duration := intervals[i].start_time - current_end) >= min_gap
        and (current_end := max(current_end, intervals[i].end_time)) is not None
    ]


def process_single_file(tg_file: str, config: configparser.ConfigParser, 
                       subset: str, speech_path: str, silence_path: str):
    """Process a single TextGrid file with optimized operations"""
    all_tiers = load_tiers(tg_file)
    silence_gaps = find_silence_gaps(all_tiers)
    
    audio_file = f"{os.path.splitext(os.path.basename(tg_file))[0]}.wav"
    audio_file = os.path.join(f"{config['DATA']['audio']}/{subset}", audio_file)
    
    cut_audio(audio_file, all_tiers, silence_gaps, speech_path, silence_path)


def cut_audio(audio_path, speech_intervals, silence_intervals, speech_path, silence_path):
    """
    cut audio for speech and silence intervals

    Args:
        audio_path (str): the path to the audio recording
        speech_intervals (list): the list of adjusted speech intervals
        silence_intervals (list): the list of silence gaps
        speech_path (str): the path to store speech audio clips
        silence_path (str): the path to store silence audio clips
    """
    audio_array = AudioSegment.from_wav(audio_path)
    base_filename = os.path.splitext(os.path.basename(audio_path))[0]
    
    # Process speech intervals
    for i, interval in enumerate(speech_intervals):
        file_name = f"{base_filename}_speech_{i}.wav"
        out_file = os.path.join(speech_path, file_name)
        start_ms = int(interval.start_time * 1000)
        end_ms = int(interval.end_time * 1000)
        segment = audio_array[start_ms:end_ms]
        segment.export(out_file, format="wav")
    
    # Process silence intervals
    for i, interval in enumerate(silence_intervals):
        file_name = f"{base_filename}_silence_{i}.wav"
        out_file = os.path.join(silence_path, file_name)
        start_ms = int(interval.start_time * 1000)
        end_ms = int(interval.end_time * 1000)
        segment = audio_array[start_ms:end_ms]
        segment.export(out_file, format="wav")


def process_single_file(tg_file, config, subset, speech_path, silence_path):
    """Process a single TextGrid file"""
    all_tiers = load_tiers(tg_file)
    silence_gaps = find_silence_gaps(all_tiers)
    
    audio_file = f"{os.path.splitext(os.path.basename(tg_file))[0]}.wav"
    audio_file = os.path.join(f"{config['DATA']['audio']}/{subset}", audio_file)
    
    cut_audio(audio_file, all_tiers, silence_gaps, speech_path, silence_path)

if __name__ == "__main__":
    pargs = parge_args()
    config = configparser.ConfigParser()
    config.read("config.ini")
    total_df = []
    output_file = "../coraal_amplitudes.csv"
    for subset in ("ATL", "DCA", "DCB", "LES", "PRV", "ROC", "VLD"):
        silence_path = f"{config['DATA']['silence']}/{subset}"
        speech_path = f"{config['DATA']['speech']}/{subset}"
        os.makedirs(silence_path, exist_ok=True)
        os.makedirs(speech_path, exist_ok=True)
        
        tg_path = f"{config['DATA']['text_grid']}/{subset}"
        tg_files = glob(f"{tg_path}/*.TextGrid")
        
        if pargs.cut_audio:
            print(f"Processing subset: {subset}")
            for tg_file in tg_files:
                process_single_file(tg_file, config, subset, speech_path, silence_path)
        else:
            if os.path.exists(output_file):
                pass
            else:
                total_df.extend(get_amplitudes(silence_path, audio_type="silence"))
                total_df.extend(get_amplitudes(speech_path, audio_type="speech"))
    if not os.path.exists(output_file):
        total_df = pd.DataFrame(total_df)
        total_df.to_csv(output_file, index=False)
    else:
        les_path = f"{config['DATA']['silence']}/LES"
        get_amplitudes(les_path, "silence", expected_sr=44100)