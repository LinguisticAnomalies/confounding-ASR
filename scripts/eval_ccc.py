from datetime import datetime
import configparser
import os
import re
from glob import glob
import argparse
import torch
import pandas as pd
import evaluate
from TRESTLE import (
    TextWrapperProcessor,
    AudioProcessor
)
from datasets import (
    load_dataset,
    Audio
)
from transformers import (
    AutoProcessor,
    WhisperForConditionalGeneration,
    WhisperTokenizer
)
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

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
    return parser.parse_args()

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
    batch['path'] = audio['path']
    batch['prediction'] = preds
    return batch


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


def get_output_file(args, group=None):
    """
    Generate the output file path based on the provided arguments and model name.

    Parameters:
        args (argparse.Namespace): Command line arguments.
        name (str): Name of the model.
        group (str): the geographic location for the speaker

    Returns:
        str: The output file path.
    """
    file_suffix = f"_{group}" if group else ""
    ft_suffix = "_ft" if args.finetune else ""
    out_file = f"../whisper-output-clean/ccc/{args.model_name}{ft_suffix}{file_suffix}.csv"
    return out_file


if __name__ == "__main__":
    start_time = datetime.now()
    pargs = parge_args()
    config = configparser.ConfigParser()
    config.read("config.ini")
    ccc_txt_patterns = {
        r"[\(\[].*?[\)\]][?!,.]?": "",
        r'\^+_+\s?[?.!,]?': "",
        r'~': "",
        r'-': ' ',
        r'[^\x00-\x7F]+': '',
        r'\<|\>': ' ',
        r"\_": "",
        r'\s+': " ",
        r'[^\w\s\']': '',
    }
    ccc_sample = {
        "format": ".TextGrid",
        "text_input_path": config['DATA']['ccc_text_input'],
        "audio_input_path": config['DATA']['ccc_audio_input'],
        "text_output_path": config['DATA']['ccc_text_output'],
        "audio_output_path": config['DATA']['ccc_audio_output'],
        "audio_type": ".wav",
        "data_type": "ccc",
    }
    if len(glob(f"{config['DATA']['ccc_text_output']}/*.jsonl")) > 0:
        if len(glob(f"{config['DATA']['ccc_audio_output']}/*.wav")) > 0:
            pass
        else:
            processor = AudioProcessor(data_loc=ccc_sample, sample_rate=16000)
            processor.process_audio()
    else:
        wrapper_processor = TextWrapperProcessor(
            data_loc=ccc_sample, txt_patterns=ccc_txt_patterns)
        wrapper_processor.process()
        processor = AudioProcessor(data_loc=ccc_sample, sample_rate=16000)
        processor.process_audio()
    pred_out_file = get_output_file(pargs)
    EPOCHS = 4
    if os.path.exists(pred_out_file):
        # read meta data and do a basic name match
        meta_df = pd.read_csv("../participants.csv")
        meta_df = meta_df.loc[meta_df['corpus_name'] == "CCC"]
        meta_df = meta_df.loc[meta_df['role'] == "interviewee"]
        meta_df = meta_df[['name', 'condition_or_disease', 'occupation', 'race', 'age']]
        meta_df['name'] = meta_df['name'].str.replace("Mr. ", "")
        meta_df['name'] = meta_df['name'].str.replace("Ms. ", "")
        meta_df['name'] = meta_df['name'].str.replace("B ", "")
        meta_df['name'] = meta_df['name'].str.replace("P ", "")
        meta_df['name'] = meta_df['name'].str.split('_|-')
        meta_df['name'] = meta_df['name'].apply(lambda x: x[0])
        meta_df['condition_or_disease'] = meta_df['condition_or_disease'].str.replace("\n", ",")
        meta_df = meta_df.drop_duplicates()
        pred_df = pd.read_csv(pred_out_file)
        pred_df = post_process(pred_df)
        # by speaker
        pred_df['path'] = pred_df['path'].str.split("/").str[-1]
        pred_df['speaker'] = pred_df['path'].str.split("_", n=2).str[0]
        pred_df = pred_df[['speaker', 'transcription', 'prediction']]
        speakers = pred_df['speaker'].unique()
        records = []
        for speaker in speakers:
            sub_df = pred_df.loc[pred_df['speaker'] == speaker]
            condition = meta_df.loc[meta_df['name'] == speaker]
            if not condition.empty:
                overall_wer = evaluate.load("wer")
                overall_cer = evaluate.load("cer")
                wer= overall_wer.compute(
                    predictions=sub_df['prediction'].values.tolist(),
                    references=sub_df['transcription'].values.tolist(),
                )
                cer = overall_cer.compute(
                    predictions=sub_df['prediction'].values.tolist(),
                    references=sub_df['transcription'].values.tolist(),
                )
                record = {
                    'name': speaker, 
                    'wer': round(wer*100, 2),
                    'cer': round(cer*100, 2)}
                records.append(record)
        wers = pd.DataFrame(records)
        output_df = wers.merge(meta_df, on='name')
        if pargs.finetune:
            output_df.to_csv(f"../{pargs.model_name}_ft_ccc.csv", index=False)
        else:
            output_df.to_csv(f"../{pargs.model_name}_ccc.csv", index=False)
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
        ccc_dt = load_dataset(
            "audiofolder", data_dir=config['DATA']['ccc_audio_output'])
        print("processing dataset...")
        ccc_dt = ccc_dt.cast_column("audio", Audio(sampling_rate=16000))
        result = ccc_dt.map(map_to_pred, remove_columns=['audio'])
        result = result['train']
        result.to_csv(pred_out_file)
    print(f"Total running time: {datetime.now()-start_time}")