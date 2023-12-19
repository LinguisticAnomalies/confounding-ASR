'''
Fine tune whisper on CORAAL test set
'''


from datetime import datetime
import warnings
import os
import argparse
import configparser
import shutil
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import pandas as pd
import torch
import evaluate
from datasets import load_dataset, Audio, Dataset
from transformers import AutoProcessor, WhisperForConditionalGeneration
from transformers import WhisperFeatureExtractor, WhisperTokenizer
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
warnings.filterwarnings('ignore')
os.environ["WANDB_PROJECT"]="coraal"


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def parge_args():
    """
    add argument parser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", required=True, type=str,
        help="""the name of whisper model"""
    )
    return parser.parse_args()


def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]
    # compute log-Mel input features from input audio array 
    batch["input_features"] = feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    # encode target text to label ids 
    batch["labels"] = tokenizer(batch["transcription"]).input_ids
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


if __name__ == "__main__":
    start_time = datetime.now()
    pargs = parge_args()
    model_card = f"openai/{pargs.model_name}"
    model_name = model_card.split("/")[-1]
    EPOCHS = 10
    config = configparser.ConfigParser()
    config.read("config.ini")
    processor = AutoProcessor.from_pretrained(
        model_card)
    tokenizer = WhisperTokenizer.from_pretrained(
        model_card)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_card)
    coraal_dt = load_dataset(
        "audiofolder", data_dir=config['DATA']['dataset'],)
        # split="train[:20%]")
    # coraal_dt = coraal_dt.train_test_split(test_size=0.3)
    coraal_dt = coraal_dt.cast_column("audio", Audio(sampling_rate=16000))
    # print("preprocessing the audio dataset")
    coraal_dt = coraal_dt.map(prepare_dataset, num_proc=16)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_card,)
        # use_flash_attention_2=True)
    model.is_parallelizable = True
    model.model_parallel = True
    model.config.use_cache = False
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="english", task="transcribe")
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    # training process
    training_args = Seq2SeqTrainingArguments(
        output_dir=f"../{model_name}", 
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        lr_scheduler_type='cosine',
        warmup_steps=100,
        # speed up
        gradient_checkpointing=True,
        fp16=True,
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        generation_max_length=225,
        num_train_epochs=EPOCHS,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        report_to="wandb",
        run_name=f"{model_name}-clean",
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        auto_find_batch_size=True,
        torch_compile=True,
        seed=42,
        data_seed=42,
        group_by_length=False
    )
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=coraal_dt["train"],
        eval_dataset=coraal_dt["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics_wer,
        tokenizer=processor.feature_extractor,
    )
    trainer.train()
    # save to local
    model.save_pretrained(f"../ft-models/{model_name}/model/")
    processor.save_pretrained(f"../ft-models/{model_name}/processor/")
    # remove checkpoints
    shutil.rmtree(f"../{model_name}")
    print(f"Total running time: {datetime.now() - start_time}")