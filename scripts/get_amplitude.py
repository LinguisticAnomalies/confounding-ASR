import os
import re
import configparser
import argparse
import shutil
from typing import List, Tuple, Optional
import time
from dataclasses import dataclass
from glob import glob
from tqdm import tqdm
import librosa
from pydub import AudioSegment, silence
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


def cut_audio(input_path, silence_path, speech_path, remove=True):
    """cut the audio recordings into silience and speech clips

    Args:
        input_path (str): the path to the original audio recordings
        silence_path (str): the path to the silence clips by subset
        speech_path (str): the path to the speech clips by subset
        remove (bool, optional): if remove the previously cut audio clips. Defaults to True.
    """
    if remove:
        if os.path.exists(silence_path):
            shutil.rmtree(silence_path)
        if os.path.exists(speech_path):
            shutil.rmtree(speech_path)
    audio_clips = glob(f"{input_path}/*.wav")
    os.makedirs(silence_path, exist_ok=True)
    os.makedirs(speech_path, exist_ok=True)
    # for more precise detection
    seek_step = 1
    # find and cut any silence clips > 100ms
    for clip in tqdm(audio_clips, desc="Finding and sliding the silences in the audio"):
        filename = os.path.basename(clip)
        base_filename = os.path.splitext(filename)[0]
        audio_array = AudioSegment.from_wav(clip)
        dBFS = audio_array.dBFS
        silences = silence.detect_silence(
                audio_array,
                min_silence_len=500,
                silence_thresh=dBFS-25,
                seek_step=seek_step)
        if silences:
            silences.sort(key=lambda x: x[0])
            audio_length = len(audio_array)
            last_end = 0
            for i, (start, end) in enumerate(silences):
                adjusted_start = min(start + seek_step, audio_length)
                adjusted_end = max(end - seek_step, 0)
                if adjusted_end > adjusted_start:
                    silence_audio = audio_array[adjusted_start:adjusted_end]
                    # double check if the silence is indeed silence
                    segment_dBFS = silence_audio.dBFS
                    if segment_dBFS is not None and segment_dBFS < dBFS-16:
                        silence_filename = f"{base_filename}_silence_{i}.wav"
                        silence_audio.export(
                            os.path.join(silence_path, silence_filename),
                            format="wav"
                        )
                if start > last_end:
                    speech_start = last_end + seek_step
                    speech_end = start - seek_step
                    if speech_end > speech_start:
                        speech_audio = audio_array[speech_start:speech_end]
                        speech_filename = f"{base_filename}_speech_{i}.wav"
                        speech_audio.export(
                            os.path.join(speech_path, speech_filename),
                            format="wav"
                        )
                last_end = end
            if last_end < audio_length - seek_step:
                speech_audio = audio_array[last_end + seek_step:audio_length]
                speech_filename = f"{base_filename}_speech_final.wav"
                speech_audio.export(
                    os.path.join(speech_path, speech_filename),
                    format="wav"
                )


def get_amplitudes(input_path, expected_sr=44100):
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
    audio_type = input_path.split("/")[-2]
    audio_files = glob(f"{input_path}/*.wav")
    for audio_file in tqdm(
        audio_files, total=len(audio_files), desc=f"Processing {audio_type} audio clips"):
        filename = os.path.basename(audio_file)
        base_filename = os.path.splitext(filename)[0]
        y, _ = librosa.load(path=audio_file, sr=expected_sr)
        db = 20*np.log10(np.abs(y) + 1e-10)
        mean_amplitudes.append(
            {"subset": component, 
             "filename": base_filename,
             "type": audio_type, 
             "amplitudes": np.mean(db)})
    return mean_amplitudes


def plot_separate_violins(df):
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
    # load tiers
    for tiers in tg.tiers:
        # include misc for silence
        for interval in tiers:
            text = interval.mark.strip()
            all_intervals.append(
                SpeakingInterval(
                    start_time=interval.minTime,
                    end_time=interval.maxTime,
                    type="speech",
                    text= text if text else None
                )
            )
    return sorted(all_intervals, key=lambda x: x.start_time)


def find_silence_gaps(intervals, min_gap=0.05):
    silence_gaps = []
    if not intervals:
        return []
    current_end = intervals[0].end_time
    for i in range(1, len(intervals)):
        current = intervals[i]
        previous = intervals[i-1]
        next_start = current.start_time
        if next_start > current_end:
            gap_duration = next_start - current_end
            if gap_duration >= min_gap:
                # if current.text is None and previous.text is None:
                print(intervals[i])
                print(intervals[i-1])
                temp_si = SpeakingInterval(
                    start_time=current_end,
                    end_time=next_start,
                    type="silence",
                    text=None)
                current_end = max(current_end, current.end_time)
                print(temp_si)
                time.sleep(3)
                print("--------------")
    return silence_gaps

if __name__ == "__main__":
    pargs = parge_args()
    config = configparser.ConfigParser()
    config.read("config.ini")
    total_df = []
    output_file = "../coraal_amplitudes.csv"
    do_calculate = True
    if os.path.exists(output_file):
        do_calculate = False
    subsets = ("ATL", "DCA", "DCB", "LES", "PRV", "ROC", "VLD")
    for subset in subsets:
        print(f"-------- {subset} -------")
        tg_path = f"{config['DATA']['text_grid']}/{subset}"
        tg_files = glob(f"{tg_path}/*.TextGrid")
        for test_file in tg_files:
            print(test_file)
            all_tiers = load_tiers(test_file)
            silence_gaps = find_silence_gaps(all_tiers)
            print(silence_gaps)
            
        break
    # for subset in subsets:
    #     silence_path = f"{config['DATA']['silence']}/{subset}"
    #     speech_path = f"{config['DATA']['speech']}/{subset}"
    #     if pargs.cut_audio:
    #         print(f"Processing CORAAL {subset} subset...")
    #         input_path = f"{config['DATA']['audio']}/{subset}"
    #         cut_audio(
    #             input_path, silence_path, speech_path)
    #     if do_calculate:
    #         total_df.extend(get_amplitudes(silence_path))
    #         total_df.extend(get_amplitudes(speech_path))
    # if do_calculate:
    #     total_df = pd.DataFrame(total_df)
    #     total_df.to_csv(output_file, index=False)
    # else:
    #     total_df = pd.read_csv(output_file)
    #     # by recording/type/subset
    #     total_df['filename'] = total_df['filename'].str.replace(r'_[^_]*_[^_]*$', '', regex=True)
    #     total_df = total_df.groupby(['subset', 'filename', 'type']).mean().reset_index()
    #     total_df.to_csv("../coraal_amlitudes_mean.csv", index=False)
    #     plot_separate_violins(total_df)
