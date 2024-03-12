'''
Evaluate pre-trained Whisper large v2 on noice-reduced CORAAL
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
        "--model_name", required=True, type=str,
        help="""the name of whisper model"""
    )
    parser.add_argument(
        "--finetune", action="store_true",
        help="""if using fine-tuned whisper model"""
    )
    parser.add_argument(
        "--location", required=True,
        type=str,
        help="""the name of the sub component of CORAAL"""
    )
    return parser.parse_args()


def create_noice_profile(input_path, loc, profile_path):
    """
    create noice profile for a sub-component of CORAAL

    :param input_path: the folder containing utterance-level audio clips
    :type input_path: str
    :param loc: the sub component name
    :type loc: str
    :param profile_path: the folder for storing nocise profiles
    :type profile_path: str
    """
    audio_clips = glob(f"{input_path}{loc}/*.wav")
    loc_subfolder = os.path.join(profile_path, loc)
    os.makedirs(loc_subfolder, exist_ok=True)
    for clip in tqdm(audio_clips, desc="Generating noise profile files"):
        clip_file = os.path.splitext(os.path.basename(clip))[0]
        clip_file = f"{clip_file}_noise.prof"
        output_clip = os.path.join(loc_subfolder, clip_file)
        os.system(f"sox {clip} -n noiseprof {output_clip}")


def reduce_noise(input_path, loc, profile_path, output_path):
    audio_clips = glob(f"{input_path}{loc}/*.wav")
    nr_subfolder = os.path.join(output_path, loc)
    os.makedirs(nr_subfolder, exist_ok=True)
    for clip in tqdm(audio_clips, desc="Reducing noise from audio clips"):
        clip_file = os.path.splitext(os.path.basename(clip))[0]
        noise_profile = f"{clip_file}_noise.prof"
        noise_profile_path = os.path.join(profile_path, loc, noise_profile)
        output_clip = os.path.join(output_path, loc, f"{clip_file}_denoised.wav")
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
    input_features = input_features.to('cuda')
    with torch.no_grad():
        predicted_ids = model.generate(input_features)
    preds = processor.batch_decode(
        predicted_ids, skip_special_tokens=True, normalize=False)[0]
    batch['prediction'] = preds
    batch['path'] = audio['path']
    return batch


def get_output_file(args):
    file_suffix = f"_{args.location}"
    ft_suffix = "_ft" if args.finetune else ""
    out_file = f"{pargs.model_name}{ft_suffix}{file_suffix}"
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
    start_time = datetime.now()
    EPOCHS = 4
    pargs = parge_args()
    loc = pargs.location
    config = configparser.ConfigParser()
    config.read("config.ini")
    val_noise_profile_path = os.path.join(config['DATA']['val_noise_profile'], loc)
    val_nr_path = os.path.join(config['DATA']['val_nr'], loc)
    if not os.path.exists(val_noise_profile_path) or not os.path.exists(val_nr_path):
        create_noice_profile(
            config['DATA']['val'], loc,
            config['DATA']['val_noise_profile'])
        reduce_noise(
            config['DATA']['val'], loc,
            config['DATA']['val_noise_profile'],
            config['DATA']['val_nr'])
    else:
        if not os.listdir(val_noise_profile_path):
            create_noice_profile(
                config['DATA']['val'], loc,
                config['DATA']['val_noise_profile'])
        if not os.listdir(val_nr_path):
            reduce_noise(
                config['DATA']['val'], loc,
                config['DATA']['val_noise_profile'],
                config['DATA']['val_nr'])
    out_file = get_output_file(pargs)
    if os.path.exists(out_file):
        denoised_df = pd.read_csv(out_file)
        if pargs.finetune:
            trans_df = pd.read_csv(
                f"../whisper-output-clean/val/{pargs.model_name}_ft_{pargs.location}_4.csv")
        else:
            trans_df = pd.read_csv(
                f"../whisper-output-clean/val/{pargs.model_name}_{pargs.location}.csv")
        trans_df['path'] = trans_df['path'].str.replace("data3", "data4")
        denoised_df = post_process(denoised_df)
        denoised_df['path'] = denoised_df['path'].str.replace("_denoised", "")
        denoised_df['path'] = denoised_df['path'].str.replace("_nr", "")
        denoised_df = denoised_df.rename(columns={'prediction': 'denoised_pred'})
        full_df = denoised_df.merge(trans_df, on='path')
        full_df = post_process(full_df)
        overall_wer = evaluate.load("wer")
        wer= overall_wer.compute(
            predictions=full_df['prediction'].values.tolist(),
            references=full_df['transcription'].values.tolist()
        )
        print(f"NO noise reduction. Fine-tuned: {pargs.finetune}. Overall WER: {round(wer*100, 2)}")
        wer= overall_wer.compute(
            predictions=full_df['denoised_pred'].values.tolist(),
            references=full_df['transcription'].values.tolist()
        )
        print(f"Noise reduction. Fine-tuned: {pargs.finetune}. Overall WER: {round(wer*100, 2)}")
    else:
        tokenizer = WhisperTokenizer.from_pretrained(
            f"openai/{pargs.model_name}", language="english", task="transcribe")
        if pargs.finetune:
            processor = AutoProcessor.from_pretrained(
                f"../ft-models/{pargs.model_name}-{EPOCHS}/processor",
                language="english", task="transcribe")
            model = WhisperForConditionalGeneration.from_pretrained(
                f"../ft-models/{pargs.model_name}-{EPOCHS}/model")
        else:
            model = WhisperForConditionalGeneration.from_pretrained(
                f"openai/{pargs.model_name}"
            )
            processor = AutoProcessor.from_pretrained(
                f"openai/{pargs.model_name}", language="english", task="transcribe")
        model.to('cuda')
        model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
            language="english", task="transcribe")
        
        sub_folder = os.path.join(config['DATA']['val_nr'], loc)
        coraal_dt = load_dataset("audiofolder", data_dir=sub_folder)
        coraal_dt = coraal_dt.cast_column("audio", Audio(sampling_rate=16000))
        result = coraal_dt.map(map_to_pred, remove_columns=['audio'])
        result = result['train']
        result.to_csv(out_file)
    print(f"Total running time: {datetime.now()-start_time}")