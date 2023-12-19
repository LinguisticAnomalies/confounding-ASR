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
    batch['prediction'] = processor.tokenizer._normalize(preds)
    batch["transcription"] = processor.tokenizer._normalize(batch['transcription'])
    return batch


def map_to_pred_with_prompt(batch):
    """
    Perform inference on an audio batch using prompting

    Parameters:
        batch (dict): A dictionary containing audio data and other related information.

    Returns:
        dict: The input batch dictionary with added prediction and transcription fields.
    """
    # need to add CORAAL conversion here
    # NOTE: this prompt here is losely to hot words in CTC decoding
    prompt_ids = processor.get_prompt_ids("Write in African American accent:")
    audio = batch['audio']
    input_features = processor(
        audio['array'], sampling_rate=audio['sampling_rate'], return_tensors="pt").input_features
    input_features = input_features.to('cuda')
    with torch.no_grad():
        predicted_ids = model.generate(input_features, prompt_ids=prompt_ids)
    batch['prediction'] = processor.batch_decode(
        predicted_ids, skip_special_tokens=True, prompt_ids=prompt_ids)[0]
    return batch


def get_output_file(args, name):
    """
    Generate the output file path based on the provided arguments and model name.

    Parameters:
        args (argparse.Namespace): Command line arguments.
        name (str): Name of the model.

    Returns:
        str: The output file path.
    """
    if args.finetune:
        out_file = f"../whisper-output/{name}_ft.json"
    else:
        if args.prompt:
            out_file = f"../whisper-output/{name}_prompt.json"
        else:
            out_file = f"../whisper-output/{name}.json"
    return out_file


if __name__ == "__main__":
    start_time = datetime.now()
    pargs = parge_args()
    config = configparser.ConfigParser()
    config.read("config.ini")
    MODEL_CARD = f"openai/{pargs.model_name}"
    MODEL_NAME = MODEL_CARD.rsplit('/', maxsplit=1)[-1]
    pred_out_file = get_output_file(pargs, MODEL_NAME)
    if os.path.exists(pred_out_file):
        # calculate WER
        pred = pd.read_json(pred_out_file, lines=True)
        pred = pred[pred['transcription'] != '']
        wer = evaluate.load("wer")
        cer = evaluate.load("cer")
        wer_metric = 100* wer.compute(
            predictions=pred['prediction'].values.tolist(), references=pred['transcription'].values.tolist())
        cer_metric = 100* cer.compute(
            predictions=pred['prediction'], references=pred['transcription'])
        print(f"{MODEL_NAME}, fine-tuned: {pargs.finetune}, prompting: {pargs.prompt}")
        print(f"WER: {round(wer_metric, 2)} \tCER: {round(cer_metric, 2)}")
        print("------")
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
        if pargs.prompt:
            result = coraal_dt.map(map_to_pred_with_prompt)
        else:
            result = coraal_dt.map(map_to_pred)
        result = result.remove_columns("audio")
        result.set_format(type="pandas", columns=['transcription', 'prediction'])
        result.to_json(pred_out_file)
    print(f"Total running time: {datetime.now()-start_time}")
