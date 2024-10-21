'''
Detect silence from the audio and use silence to create noise profile
'''
from tqdm import tqdm
from datetime import datetime
from glob import glob
import warnings
import os
import re
import configparser
import argparse
import torch
import evaluate
from pydub import AudioSegment, silence
import pandas as pd
from datasets import load_dataset, Audio
from transformers import (
    AutoProcessor,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
)
warnings.filterwarnings('ignore')


def parge_args():
    """
    add argument parser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--finetune", action="store_true",
        help="""if using fine-tuned whisper model"""
    )
    return parser.parse_args()


def create_profile(input_path, profile_path):
    """find the longest silence that is at least 500ms,
        and create a noise profile for future noise reduction

    Args:
        input_path (str): the folder with audio clips
        profile_path (str): the location for saving noise profile
    """
    audio_clips = glob(f"{input_path}/*.wav")
    os.makedirs(profile_path, exist_ok=True)
    max_silence = 0
    for clip in tqdm(audio_clips, desc="Finding the silences in the audio"):
        audio_array = AudioSegment.from_wav(clip)
        dBFS = audio_array.dBFS
        silences = silence.detect_silence(
            audio_array, min_silence_len=500, silence_thresh=dBFS-16)
        if silences:
            for silence_interval in silences:
                curr_silence  = silence_interval[1] - silence_interval[0]
                if curr_silence > max_silence:
                    print("Found the longest silence")
                    max_silence = curr_silence
                    clip_file = os.path.basename(clip).split(".")[0].split("_")[0]
                    # write silence period to a local file
                    sliced_audio = audio_array[silence_interval[0]:silence_interval[1]]
                    # create noise profile
                    silence_file = os.path.join(profile_path, f"{clip_file}.wav")
                    sliced_audio.export(silence_file, format='wav')
                    output_clip = os.path.join(profile_path, f"{clip_file}_noise.prof")
                    os.system(f"sox {silence_file} -n noiseprof {output_clip}")
                    # remove silence audio
                    os.remove(os.path.join(profile_path, f"{clip_file}.wav"))


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
        clip_file = os.path.basename(clip).split(".")[0]
        loc_indicator = os.path.basename(clip).split(".")[0].split("_")[0]
        noise_profile = f"{loc_indicator}_noise.prof"
        noise_profile_path = os.path.join(profile_path, noise_profile)
        output_clip = os.path.join(output_path, f"{clip_file}_denoised.wav")
        os.system(f"sox {clip} {output_clip} noisered {noise_profile_path} 0.21")


def map_to_pred(batch):
    """
    Perform inference on an audio batch

    Parameters:
        batch (dict): A dictionary containing audio data and other related information.

    Returns:
        dict: The input batch dictionary with added prediction and transcription fields.
    """
    audio = batch['audio']
    input_features = processor(
        audio['array'], sampling_rate=audio['sampling_rate'], return_tensors="pt").input_features
    input_features = input_features.to(device)
    with torch.no_grad():
        predicted_ids = model.generate(input_features)
    preds = processor.batch_decode(
        predicted_ids, skip_special_tokens=True, normalize=False)[0]
    batch['prediction'] = preds
    batch['path'] = audio['path']
    return batch

def get_output_file(args, loc):
    file_suffix = f"_{loc}"
    ft_suffix = "_ft" if args.finetune else ""
    out_file = f"whisper-large-v2{ft_suffix}{file_suffix}"
    out_file = f"../whisper-output-clean/val_nr/{out_file}.csv"
    return out_file


def post_process(input_df):
    input_df['prediction'] = input_df["prediction"].str.lower()
    input_df['prediction'] = input_df["prediction"].apply(
        lambda x: re.sub(r'[^a-zA-Z0-9\s\']+$', '', str(x)))
    input_df['prediction'] = input_df['prediction'].str.strip()
    if 'transcription' in input_df.columns:
        input_df['transcription'] = input_df["transcription"].str.lower()
        input_df['transcription'] = input_df["transcription"].apply(
            lambda x: re.sub(r'[^a-zA-Z0-9\s\']+$', '', str(x)))
        input_df['transcription'] = input_df['transcription'].str.strip()
    return input_df


if __name__ == "__main__":
    pargs = parge_args()
    config = configparser.ConfigParser()
    config.read("config.ini")
    locs = ("ATL", "DCA", "DCB", "LES", "PRV", "ROC", "VLD")
    device = "cuda:3"
    for loc in locs:
        print(f"----- Process {loc} ------")
        subset_input = os.path.join(config['DATA']['val'], loc)
        subset_np = os.path.join(config['DATA']["val_noise_profile"], loc)
        subset_nr = os.path.join(config['DATA']["val_nr"], loc)
        if not os.path.exists(os.path.join(subset_np, f"{loc}_noise.prof")):
            create_profile(subset_input, subset_np)
            reduce_noise(subset_input, subset_np, subset_nr)
        out_file = get_output_file(pargs, loc)
        if os.path.exists(out_file):
            overall_wer = evaluate.load("wer")
            output_df = pd.read_csv(out_file)
            output_df = post_process(output_df)
            wer= overall_wer.compute(
                predictions=output_df['prediction'].values.tolist(),
                references=output_df['transcription'].values.tolist()
            )
            if pargs.finetune:
                print(f"Noise reduced, fine-tuned: WER - {round(wer*100, 2)}")
            else:
                print(f"Noise reduced, pre-trained: WER - {round(wer*100, 2)}")
        else:
            tokenizer = WhisperTokenizer.from_pretrained(
                "openai/whisper-large-v2", language="english", task="transcribe")
            if pargs.finetune:
                processor = AutoProcessor.from_pretrained(
                    f"{config['MODEL']['checkpoint']}/whisper-large-v2-4/processor",
                    language="english", task="transcribe")
                model = WhisperForConditionalGeneration.from_pretrained(
                    f"{config['MODEL']['checkpoint']}/whisper-large-v2-4/model")
            else:
                model = WhisperForConditionalGeneration.from_pretrained(
                    "openai/whisper-large-v2"
                )
                processor = AutoProcessor.from_pretrained(
                    "openai/whisper-large-v2", language="english", task="transcribe")
            model.to(device)
            model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
            language="english", task="transcribe")

            sub_folder = os.path.join(config['DATA']['val'], loc)
            coraal_dt = load_dataset("audiofolder", data_dir=sub_folder)
            coraal_dt = coraal_dt.cast_column("audio", Audio(sampling_rate=16000))
            result = coraal_dt.map(map_to_pred, remove_columns=['audio'])
            result = result['train']
            result.to_csv(out_file)