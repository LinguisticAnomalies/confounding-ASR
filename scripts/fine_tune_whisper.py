'''
Fine tune whisper on CORAAL test set
'''
from datetime import datetime
import warnings
import argparse
import configparser
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch
import evaluate
from datasets import load_dataset, Audio
from transformers import (
    AutoProcessor,
    AutoModelForSpeechSeq2Seq,
    WhisperFeatureExtractor,
    WhisperTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed
)
warnings.filterwarnings('ignore')

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
        "--nr", action="store_true",
        help="""if fine-tuning on the noise-reduced set"""
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
    label_str = tokenizer.batch_decode(
        label_ids, skip_special_tokens=True, normalize=False)
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


if __name__ == "__main__":
    start_time = datetime.now()
    set_seed(42)
    torch_dtype = torch.float16
    pargs = parge_args()
    model_card = "openai/whisper-large-v2"
    model_name = "whisper-large-v2"
    EPOCHS = 4
    config = configparser.ConfigParser()
    config.read("config.ini")
    if pargs.nr:
        coraal_dt = load_dataset(
            "audiofolder",
            data_dir=config['DATA']['denoise'],
        )
        model_output = f"{config['MODEL']['checkpoint']}/{model_name}-{EPOCHS}-nr/model/"
        processor_output = f"{config['MODEL']['checkpoint']}/{model_name}-{EPOCHS}-nr/processor/"
    else:
        coraal_dt = load_dataset(
            "audiofolder",
            data_dir=config['DATA']['data'],
        )
        model_output = f"{config['MODEL']['checkpoint']}/{model_name}-{EPOCHS}/model/"
        processor_output = f"{config['MODEL']['checkpoint']}/{model_name}-{EPOCHS}/processor/"
    processor = AutoProcessor.from_pretrained(
        model_card)
    tokenizer = WhisperTokenizer.from_pretrained(
        model_card)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_card)
    coraal_dt = coraal_dt.cast_column("audio", Audio(sampling_rate=16000))
    print("preprocessing the audio dataset")
    coraal_dt = coraal_dt.map(
        prepare_dataset, num_proc=64, remove_columns=['audio', 'transcription'])
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_card,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        attn_implementation="sdpa",)
        #torch_dtype=torch_dtype)
    # freeze encoder
    model.freeze_encoder()
    model.is_parallelizable = True
    model.model_parallel = True
    model.config.use_cache = False
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="en", task="transcribe")
    # model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    # training process
    training_args = Seq2SeqTrainingArguments(
        output_dir=f"../{model_name}", 
        per_device_train_batch_size=32,
        gradient_accumulation_steps=1,
        learning_rate=5e-4,
        warmup_steps=100,
        # speed up
        fp16=True,
        gradient_checkpointing=True,
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        generation_max_length=225,
        num_train_epochs=EPOCHS,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        report_to="none",
        # run_name=f"{model_name}-clean-{EPOCHS}",
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        seed=42,
        # data_seed=42,
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
    model.save_pretrained(model_output)
    processor.save_pretrained(processor_output)
    print(f"Total running time: {datetime.now() - start_time}")