import os
from functools import lru_cache
import configparser
import argparse
from tqdm.contrib.concurrent import process_map
from functools import partial
from typing import List, Tuple, Optional
from dataclasses import dataclass
from glob import glob
import librosa
from pydub import AudioSegment
import numpy as np
import pandas as pd
import seaborn as sns
import textgrid
import matplotlib.pyplot as plt


@dataclass
class AudioInterval:
    start_time: float
    end_time: float


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


def process_audio_file(audio_file, component, audio_type, expected_sr=None):
    """
    Process a single audio file to get its mean amplitude.
    
    Args:
        audio_file (str): Path to audio file
        component (str): Component name
        audio_type (str): Type of audio (silence/speech)
        expected_sr (int, optional): Expected sample rate
        
    Returns:
        dict: Dictionary containing audio file metadata and amplitude
    """
    filename = os.path.basename(audio_file)
    base_filename = os.path.splitext(filename)[0]
    
    # Use memory efficient loading with dtype=np.float32
    y, _ = librosa.load(path=audio_file, sr=expected_sr, dtype=np.float32)
    
    # Vectorized computation
    db = 20 * np.log10(np.abs(y) + 1e-10)
    return {
        "subset": component,
        "filename": base_filename,
        "type": audio_type,
        "amplitudes": np.mean(db)
    }


def get_amplitudes(input_path, audio_type, n_workers=4):
    """
    Get mean amplitude of each audio clip given a folder using parallel processing.
    
    Args:
        input_path (str): Folder containing audio clips
        audio_type (str): Type of audio (silence/speech)
        n_workers (int): Number of parallel workers
        
    Returns:
        list: List of dictionaries containing amplitude data
    """
    component = input_path.split("/")[-1]
    audio_files = glob(f"{input_path}/*.wav")
    
    if not audio_files:
        return []
        
    expected_sr = None if audio_type == "silence" else 44100
    
    # get chunk size
    n_files = len(audio_files)
    chunk_size = max(1, min(
        n_files // (n_workers * 4),
        100
    ))
    
    # partial function with fixed arguments
    process_func = partial(
        process_audio_file,
        component=component,
        audio_type=audio_type,
        expected_sr=expected_sr
    )
    
    # parallel processing
    return process_map(
        process_func,
        audio_files,
        max_workers=n_workers,
        chunksize=chunk_size,
        desc=f"Processing {audio_type} audio clips"
    )


def plot_separate_violins(df):
    """violin plot for each subset in CORAAL


    Args:
        df (pandas.DataFrame): dataframe containing amplitidues for each clip
    """
    df = df.dropna()
    # drop last 2 items in the filename
    df['filename'] = df['filename'].str.split("_").str[:-2].str.join("_")
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), height_ratios=[1, 1, 1])
    silence_stats = df[df['type'] == 'silence'].groupby('subset')['amplitudes'].agg(['mean', 'std', 'min'])  # Changed max to min
    speech_stats = df[df['type'] == 'speech'].groupby('subset')['amplitudes'].agg(['mean', 'std', 'min'])   # Changed max to min
    # average SNR per location
    pivot_df = df.pivot_table(
        index=['subset', "filename"],
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
    offset = (ax1.get_ylim()[1] - ax1.get_ylim()[0]) * -0.02
    for idx, subset in enumerate(silence_stats.index):
        stats = silence_stats.loc[subset]
        ax1.text(idx, stats['min'] + offset,  # Changed max to min
                f'mean={stats["mean"]:.1f}\nstd={stats["std"]:.1f}', 
                horizontalalignment='center',
                verticalalignment='top',
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
    offset = (ax2.get_ylim()[1] - ax2.get_ylim()[0]) * -0.02
    for idx, subset in enumerate(speech_stats.index):
        stats = speech_stats.loc[subset]
        ax2.text(idx, stats['min'] + offset,  # Changed max to min
                f'mean={stats["mean"]:.1f}\nstd={stats["std"]:.1f}', 
                horizontalalignment='center',
                verticalalignment='top',
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
    min_snr = pivot_df.groupby('subset')['snr'].min()
    offset = (ax3.get_ylim()[1] - ax3.get_ylim()[0]) * -0.02
    for idx, row in snr_by_location.iterrows():
        ax3.text(idx, min_snr[row['subset']] + offset, 
                f'mean={row["mean"]:.1f}\nstd={row["std"]:.1f}', 
                horizontalalignment='center',
                verticalalignment='top',
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
    silence_gaps = []
    if not intervals:
        return []
    current_end = intervals[0].end_time
    for i in range(1, len(intervals)):
        current = intervals[i]
        next_start = current.start_time
        if next_start > current_end:
            gap_duration = next_start - current_end
            if gap_duration >= min_gap:
                silence_gaps.append(SpeakingInterval(
                    start_time=current_end,
                    end_time=next_start,
                    type="silence",
                    text=None)) 
                current_end = max(current_end, current.end_time)
    return silence_gaps


def load_audio(audio_path: str) -> tuple[AudioSegment, str]:
    """
    Load audio file and extract base filename.
    
    Args:
        audio_path: Path to the audio file
        
    Returns:
        Tuple of (AudioSegment, base_filename)
    """
    base_filename = os.path.splitext(os.path.basename(audio_path))[0]
    audio_array = AudioSegment.from_wav(audio_path)
    return audio_array, base_filename


def get_output_path(base_path: str, base_filename: str, interval_type: str, index: int) -> str:
    """
    Generate output file path for an audio segment.
    
    Args:
        base_path: Directory to store the output
        base_filename: Base name for the output file
        interval_type: Type of interval (speech/silence)
        index: Index of the interval
        
    Returns:
        Complete output file path
    """
    file_name = f"{base_filename}_{interval_type}_{index}.wav"
    return os.path.join(base_path, file_name)


def extract_segment(audio: AudioSegment, interval: AudioInterval) -> AudioSegment:
    """
    Extract a segment from audio based on time interval.
    
    Args:
        audio: Source audio
        interval: Time interval to extract
        
    Returns:
        Extracted audio segment
    """
    start_ms = int(interval.start_time * 1000)
    end_ms = int(interval.end_time * 1000)
    return audio[start_ms:end_ms]


def save_segments(audio: AudioSegment,
                 intervals: List[AudioInterval],
                 output_path: str,
                 base_filename: str,
                 interval_type: str) -> None:
    """
    Save multiple audio segments to files.
    
    Args:
        audio: Source audio
        intervals: List of time intervals
        output_path: Directory to store output files
        base_filename: Base name for output files
        interval_type: Type of intervals (speech/silence)
    """
    for i, interval in enumerate(intervals):
        out_file = get_output_path(output_path, base_filename, interval_type, i)
        segment = extract_segment(audio, interval)
        segment.export(out_file, format="wav")


def process_single_file(tg_file: str, config: configparser.ConfigParser, 
                       subset: str, speech_path: str, silence_path: str):
    """Process a single TextGrid file with optimized operations"""
    all_tiers = load_tiers(tg_file)
    silence_gaps = find_silence_gaps(all_tiers)
    
    audio_file = f"{os.path.splitext(os.path.basename(tg_file))[0]}.wav"
    audio_file = os.path.join(f"{config['DATA']['audio']}/{subset}", audio_file)
    
    cut_audio(audio_file, all_tiers, silence_gaps, speech_path, silence_path)


def cut_audio(audio_path: str,
             speech_intervals: List[AudioInterval],
             silence_intervals: List[AudioInterval],
             speech_path: str,
             silence_path: str) -> None:
    """
    Cut audio into speech and silence segments.
    
    Args:
        audio_path: Path to the audio recording
        speech_intervals: List of speech intervals
        silence_intervals: List of silence gaps
        speech_path: Path to store speech audio clips
        silence_path: Path to store silence audio clips
    """
    audio, base_filename = load_audio(audio_path)
    # speech segments
    save_segments(audio, speech_intervals, speech_path, base_filename, "speech")
    # silence segments
    save_segments(audio, silence_intervals, silence_path, base_filename, "silence")


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
        print(f"Processing subset: {subset}")
        silence_path = f"{config['DATA']['silence']}/{subset}"
        speech_path = f"{config['DATA']['speech']}/{subset}"
        os.makedirs(silence_path, exist_ok=True)
        os.makedirs(speech_path, exist_ok=True)
        
        tg_path = f"{config['DATA']['text_grid']}/{subset}"
        tg_files = glob(f"{tg_path}/*.TextGrid")
        
        if pargs.cut_audio:
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
        total_df = pd.read_csv(output_file)
        plot_separate_violins(total_df)