'''
Transcribe CORAAL corpus using whisper model
'''


from datetime import datetime
from glob import glob
import warnings
import os
import argparse
import configparser
import torch
import pandas as pd
import evaluate
from datasets import load_dataset, Audio
from transformers import AutoProcessor, WhisperForConditionalGeneration
from transformers import WhisperFeatureExtractor, WhisperTokenizer
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
        "--prompt", action="store_true",
        help="""if using prompt for whisper transcribing"""
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
    preds = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    batch['path'] = audio['path']
    batch['prediction'] = processor.tokenizer._normalize(preds)
    batch["transcription"] = processor.tokenizer._normalize(batch['transcription'])
    return batch


def get_output_file(args, name, group=None):
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
    prompt_suffix = "_prompt" if args.prompt else ""

    out_file = f"../whisper-output-clean/{name}{ft_suffix}{file_suffix}{prompt_suffix}.csv"
    return out_file


if __name__ == "__main__":
    start_time = datetime.now()
    torch.manual_seed(42)
    pargs = parge_args()
    config = configparser.ConfigParser()
    config.read("config.ini")
    MODEL_CARD = f"openai/{pargs.model_name}"
    MODEL_NAME = MODEL_CARD.rsplit('/', maxsplit=1)[-1]
    pred_out_file = get_output_file(pargs, MODEL_NAME)
    if os.path.exists(pred_out_file):
        pass
    else:
        tokenizer = WhisperTokenizer.from_pretrained(
        MODEL_CARD, language="english", task="transcribe")
        feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_CARD)
        if pargs.finetune:
            processor = AutoProcessor.from_pretrained(f"../ft-models/{MODEL_NAME}/processor")
            model = WhisperForConditionalGeneration.from_pretrained(f"../ft-models/{MODEL_NAME}/model")
        else:
            model = WhisperForConditionalGeneration.from_pretrained(MODEL_CARD)
            processor = AutoProcessor.from_pretrained(
                MODEL_CARD, language="english", task="transcribe")
        model.to('cuda')
        model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")
        print("load dataset....")
        # load dataset
        coraal_dt = load_dataset("audiofolder", data_dir=config['DATA']['dataset'], split="test")
        print("processing dataset...")
        coraal_dt = coraal_dt.cast_column("audio", Audio(sampling_rate=16000))
        # inference
        print(f"inferencing with {MODEL_NAME}...")
        result = coraal_dt.map(map_to_pred)
        result = result.remove_columns("audio")
        result.to_csv(pred_out_file)
    print(f"Total running time: {datetime.now()-start_time}")
