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
import evaluate
from datasets import load_dataset, Audio, Dataset
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


def compute_metrics_wer(pred):
    metric = evaluate.load('wer')
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(
        pred_ids, skip_special_tokens=True, normalize=False)
    label_str = tokenizer.batch_decode(
        label_ids, skip_special_tokens=True, normalize=False)
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


def get_output_file(args, name, epoch, group=None):
    file_suffix = f"_{group}" if group else ""
    ft_suffix = "_ft" if args.finetune else ""
    out_file = f"{name}{ft_suffix}{file_suffix}_{epoch}.csv"
    out_file = f"../whisper-output-clean/val/{name}{ft_suffix}{file_suffix}_{epoch}.csv"
    return out_file


if __name__ == "__main__":
    start_time = datetime.now()
    torch.manual_seed(42)
    pargs = parge_args()
    config = configparser.ConfigParser()
    config.read("config.ini")
    # number of epochs for fine-tuning
    EPOCHS = 4
    if pargs.model_name.startswith("whisper"):
        MODEL_CARD = f"openai/{pargs.model_name}"
    else:
        MODEL_CARD = f"distil-whisper/{pargs.model_name}"
    MODEL_NAME = MODEL_CARD.rsplit('/', maxsplit=1)[-1]
    tokenizer = WhisperTokenizer.from_pretrained(
    MODEL_CARD, language="english", task="transcribe")
    if pargs.finetune:
        processor = AutoProcessor.from_pretrained(
            f"../ft-models/{MODEL_NAME}-{EPOCHS}/processor")
        model = WhisperForConditionalGeneration.from_pretrained(
            f"../ft-models/{MODEL_NAME}-{EPOCHS}/model")
    else:
        model = WhisperForConditionalGeneration.from_pretrained(MODEL_CARD)
        processor = AutoProcessor.from_pretrained(
            MODEL_CARD, language="english", task="transcribe")
    model.to('cuda')
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="english", task="transcribe")
    print("load dataset....")
    # load the dataset by geographic location
    os.makedirs("../whisper-output-clean/val", exist_ok=True)
    vali_set = glob(f"{config['DATA']['val']}/*")
    for sub_folder in vali_set:
        geo_group = sub_folder.split("/")[-1]
        print(f"inferencing on {geo_group} validation transcripts...")
        pred_out_file = get_output_file(
            pargs, MODEL_NAME, EPOCHS, group=geo_group)
        if os.path.exists(pred_out_file):
        # calculate WER
            pass
        else:
            coraal_dt = load_dataset("audiofolder", data_dir=sub_folder)
            coraal_dt = coraal_dt.cast_column("audio", Audio(sampling_rate=16000))
            result = coraal_dt.map(map_to_pred, remove_columns=['audio'])
            result = result['train']
            result.to_csv(pred_out_file)
    print(f"Total running time: {datetime.now()-start_time}")