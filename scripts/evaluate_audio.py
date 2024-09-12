'''
Evaluate audio quality by calculating the signal-to-noise ratio
'''
import os
import warnings
import configparser
from glob import glob
from tqdm import tqdm
from pydub import AudioSegment
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')


def create_noice_profile(input_path, profile_path):
    """
    create noice profile for a sub-component of CORAAL

    :param input_path: the folder containing utterance-level audio clips
    :type input_path: str
    :param profile_path: the folder for storing nocise profiles
    :type profile_path: str
    """
    audio_clips = glob(f"{input_path}/*.wav")
    os.makedirs(profile_path, exist_ok=True)
    for clip in tqdm(audio_clips, desc="Generating noise profile files"):
        clip_file = os.path.splitext(os.path.basename(clip))[0]
        clip_file = f"{clip_file}_noise.prof"
        output_clip = os.path.join(profile_path, clip_file)
        os.system(f"sox {clip} -n noiseprof {output_clip}")


def reduce_noise(input_path, profile_path, output_path):
    """
    reduce noise and save the audio clip to the local folder

    :param input_path: the folder containing original utterance-level audio clips
    :type input_path: str
    :param profile_path: the folder containing the noise profiles
    :type profile_path: str
    :param output_path: the folder storing the noise-reduced audio clips
    :type output_path: str
    """
    audio_clips = glob(f"{input_path}/*.wav")
    os.makedirs(output_path, exist_ok=True)
    for clip in tqdm(audio_clips, desc="Reducing noise from audio clips"):
        clip_file = os.path.splitext(os.path.basename(clip))[0]
        noise_profile = f"{clip_file}_noise.prof"
        noise_profile_path = os.path.join(profile_path, noise_profile)
        output_clip = os.path.join(output_path, f"{clip_file}_denoised.wav")
        os.system(f"sox {clip} {output_clip} noisered {noise_profile_path} 0.21")


def calculate_snr(original_file, reduced_file):
    """
    calculate snr given a pair of original and noise-reduced audio clip

    :param original_file: the full path to the original audio clip
    :type original_file: str
    :param reduced_file: the full path to the noise-reduced audio clip
    :type reduced_file: str
    :return: the signal noise ratio
    :rtype: np.float
    """
    original = AudioSegment.from_file(original_file)
    reduced = AudioSegment.from_file(reduced_file)
    # to numpy array
    def audiosegment_to_ndarray(audio):
        samples = audio.get_array_of_samples()
        return np.array(samples)

    original_array = audiosegment_to_ndarray(original)
    reduced_array = audiosegment_to_ndarray(reduced)
    # same length
    min_length = min(len(original_array), len(reduced_array))
    original_array = original_array[:min_length]
    reduced_array = reduced_array[:min_length]
    # calculate noise
    noise = original_array - reduced_array
    # calculate signal and noise power
    signal_power = np.mean(reduced_array**2)
    noise_power = np.mean(noise**2)
    # calculate SNR
    snr = 10 * np.log10(signal_power / noise_power)
    return snr


def batch_calculate_snr(input_path, nr_path):
    """
    SNR calculation driver

    :param input_path: _description_
    :type input_path: _type_
    :param nr_path: _description_
    :type nr_path: _type_
    :return: _description_
    :rtype: _type_
    """
    snr_results = []
    original_files = glob(f"{input_path}/*.wav")
    for file in tqdm(original_files, total=len(original_files), desc="Calculating SNR"):
        file_name = os.path.splitext(os.path.basename(file))[0]
        reduced_file = f"{file_name}_denoised.wav"
        reduced_path = os.path.join(nr_path, reduced_file)
        snr = calculate_snr(file, reduced_path)
        snr_results.append({"file": file_name, "snr": snr})
    return snr_results


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("config.ini")
    total_snr = []
    for subset in ("train", "test", "val"):
        if subset == "val":
            locs = ("ATL", "DCA", "DCB", "LES", "PRV", "ROC", "VLD")
            for loc in locs:
                subset_input = os.path.join(config['DATA'][f'{subset}'], loc)
                subset_np = os.path.join(config['DATA'][f"{subset}_noise_profile"], loc)
                subset_nr = os.path.join(config['DATA'][f"{subset}_nr"], loc)
        else:
            subset_input = config['DATA'][f'{subset}']
            subset_np = config['DATA'][f"{subset}_noise_profile"]
            subset_nr = config['DATA'][f"{subset}_nr"]
        if not os.path.exists(subset_np) or not os.path.exists(subset_nr):
            create_noice_profile(subset_input, subset_np)
            reduce_noise(subset_input, subset_np, subset_nr)
        else:
            subset_snr = batch_calculate_snr(subset_input, subset_nr)
            total_snr += subset_snr
    snr_df = pd.DataFrame(total_snr)
    snr_df.to_csv("../total_snr.csv", index=False)