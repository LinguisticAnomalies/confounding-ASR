'''
transcribe the validation set of CORAAL
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
from datasets import load_dataset, Audio, Dataset
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


def compute_metrics_wer(pred):
    metric = evaluate.load('wer')
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


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


def compute_metrics_wer(pred):
    metric = evaluate.load('wer')
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


def compute_metrics_cer(pred):
    metric = evaluate.load('cer')
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    cer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"cer": cer}


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

    out_file = f"../whisper-output/vali/{name}{ft_suffix}{file_suffix}{prompt_suffix}.json"
    return out_file


if __name__ == "__main__":
    start_time = datetime.now()
    pargs = parge_args()
    config = configparser.ConfigParser()
    config.read("config.ini")
    MODEL_CARD = f"openai/{pargs.model_name}"
    MODEL_NAME = MODEL_CARD.rsplit('/', maxsplit=1)[-1]
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
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="english", task="transcribe")
    print("load dataset....")
    # load the dataset by geographic location
    vali_set = glob(f"{config['DATA']['vali']}/*")
    for sub_folder in vali_set:
        geo_group = sub_folder.split("/")[-1]
        print(f"inferencing on {geo_group} validation transcripts...")
        pred_out_file = get_output_file(pargs, MODEL_NAME, group=geo_group)
        if os.path.exists(pred_out_file):
        # calculate WER
            pred = pd.read_json(pred_out_file, lines=True)
            pred = pred[pred['transcription'] != '']
            wer = evaluate.load("wer")
            cer = evaluate.load("cer")
            wer_metric = 100* wer.compute(
                predictions=pred['prediction'].values.tolist(),
                references=pred['transcription'].values.tolist())
            cer_metric = 100* cer.compute(
                predictions=pred['prediction'],
                references=pred['transcription'])
            print(f"{MODEL_NAME}, fine-tuned: {pargs.finetune}, prompting: {pargs.prompt}")
            print(f"{geo_group} WER: {round(wer_metric, 2)} \tCER: {round(cer_metric, 2)}")
            print("------")
        else:
            coraal_dt = load_dataset("audiofolder", data_dir=sub_folder,)
            coraal_dt = coraal_dt.cast_column("audio", Audio(sampling_rate=16000))
            result = coraal_dt.map(map_to_pred)
            result = result.remove_columns("audio")
            result = result['train']
            result.set_format(type="pandas", columns=['transcription', 'prediction'])
            result.to_json(pred_out_file)
    print(f"Total running time: {datetime.now()-start_time}")