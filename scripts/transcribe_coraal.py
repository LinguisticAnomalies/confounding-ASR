'''
Transcribe CORAAL corpus using whisper model
'''


from datetime import datetime
import warnings
import os
import argparse
import configparser
import torch
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


def get_output_file(args, name, epoch, group=None):
    file_suffix = f"_{group}" if group else ""
    ft_suffix = "_ft" if args.finetune else ""
    out_file = f"{name}{ft_suffix}{file_suffix}_{epoch}.csv"
    out_file = f"../whisper-output-clean/{out_file}"
    return out_file


if __name__ == "__main__":
    start_time = datetime.now()
    torch.manual_seed(42)
    pargs = parge_args()
    config = configparser.ConfigParser()
    config.read("config.ini")
    # number of epochs for fine-tuning
    EPOCHS = 4
    MODEL_CARD = f"openai/{pargs.model_name}"
    MODEL_NAME = MODEL_CARD.rsplit('/', maxsplit=1)[-1]
    pred_out_file = get_output_file(pargs, MODEL_NAME, EPOCHS)
    if os.path.exists(pred_out_file):
        pass
    else:
        tokenizer = WhisperTokenizer.from_pretrained(
        MODEL_CARD, language="english", task="transcribe")
        if pargs.finetune:
            processor = AutoProcessor.from_pretrained(
                f"{config['MODEL']['checkpoint']}/{MODEL_NAME}-{EPOCHS}/processor")
            model = WhisperForConditionalGeneration.from_pretrained(
                f"{config['MODEL']['checkpoint']}/{MODEL_NAME}-{EPOCHS}/model")
        else:
            model = WhisperForConditionalGeneration.from_pretrained(MODEL_CARD)
            processor = AutoProcessor.from_pretrained(
                MODEL_CARD, language="english", task="transcribe")
        model.to('cuda:2')
        model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
            language="english", task="transcribe")
        print("load dataset....")
        # load dataset
        coraal_dt = load_dataset(
            "audiofolder", data_dir=config['DATA']['data'], split="test")
        print("processing dataset...")
        coraal_dt = coraal_dt.cast_column("audio", Audio(sampling_rate=16000))
        # inference
        print(f"inferencing with {MODEL_NAME}...")
        result = coraal_dt.map(map_to_pred)
        result = result.remove_columns("audio")
        result.to_csv(pred_out_file)
    print(f"Total running time: {datetime.now()-start_time}")
