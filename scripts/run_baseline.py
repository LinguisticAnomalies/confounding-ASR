'''
Get the baseline from the whisper paper
'''


from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from glob import glob
import os
import json
import re
import whisper
import configparser
import pandas as pd
from transformers import pipeline
import evaluate
from transformers.models.whisper.english_normalizer import BasicTextNormalizer


def clean_txt(txtin):
    """
    text preprocessing for utterances

    :param txtin: the utterance
    :type txtin: str
    :return: the cleaned utterance
    :rtype: str
    """
    # remove filling words
    txtout = re.sub(r"(?i)\b(?:(?:um(?:-|h)?|nuh-|mm(?:-|mm)?)|(?:^|\W)uh\b)", "", txtin)
    # ooh -> oh
    txtout = re.sub(r"(?i)ooh", "oh", txtout)
    txtout = re.sub(r'\([^)]*\)', "", txtout)
    txtout = re.sub(r"\/[^)]*\/", "", txtout)
    txtout = re.sub(r"\<[^)]*\>", "", txtout)
    txtout = re.sub(r'[^a-zA-Z0-9\s\'-]', "", txtout)
    return txtout.lower().strip()

def clean_output(txtin):
    """
    text postprocessing for whisper-generated transcripts

    :param txtin: the whisper-generated transcript
    :type txtin: str
    :return: post-processed transcript
    :rtype: str
    """
    txtout = re.sub(r'[^a-zA-Z0-9\s\'-]', "", txtin)
    return txtout.lower().strip()


def get_files(parent_folder):
    """
    get meta files for each component

    :param parent_folder: the parent folder path to all data
    :type parent_folder: str
    """
    subfolders = ("ATL", "DCA", "DCB",
                  "PRV", "ROC", "LES", "VLD")
    for subfolder in subfolders:
        pair_dict = {}
        sub_coraal_wav = glob(os.path.join(parent_folder, "audio", subfolder, '*.wav'))
        sub_coraal_txt = glob(os.path.join(parent_folder, "txt", subfolder, '*.txt'))
        for wav_file in sub_coraal_wav:
            base_name = os.path.splitext(os.path.basename(wav_file))[0]
            matching_txt_file = next((txt for txt in sub_coraal_txt if os.path.splitext(os.path.basename(txt))[0] == base_name), None)
            if matching_txt_file:
                pair_dict[base_name] = {'wav': wav_file, 'txt': matching_txt_file}
        os.makedirs("../coraal-files/", exist_ok=True)
        with open(f"../coraal-files/{subfolder}.json", "w") as json_file:
            json.dump(pair_dict, json_file)


def get_longform_hf(info_file):
    """
    get longform transcript using HuggingFace's pipeline

    :param info_file: _description_
    :type info_file: _type_
    """
    wer = evaluate.load("wer")
    preds = []
    trans = []
    loc = os.path.splitext(os.path.basename(info_file))[0]
    normalizer = BasicTextNormalizer()
    out_file = f"../whisper-output/{loc}-hf.json"
    if os.path.exists(out_file):
        pass
    else:
        pipe = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-large-v2",
            chunk_length_s=30,
            device="cuda"
        )
        with open(info_file, "r") as json_file:
            info_obj = list(json.load(json_file).values())
            wav_files = [item['wav'] for item in info_obj]
            text_files = [item['txt'] for item in info_obj]
            for wav, text in zip(wav_files, text_files):
                tran_df = pd.read_csv(text, sep="\t")
                # NOTE: it indeed includes the talking clips from interviewees
                # tran_df = tran_df[~tran_df['Spkr'].str.contains(r'(int|misc)', case=False)]
                tran_df['clean_tran'] = tran_df["Content"].apply(clean_txt)
                tran_df.replace('', pd.NA, inplace=True)
                tran_df.dropna(inplace=True)
                text = ". ".join(tran_df["clean_tran"].values.tolist())
                pred = pipe(wav, batch_size=64)['text']
                pred = clean_output(pred)
                text = normalizer(text)
                pred = normalizer(pred)
                trans.append(text)
                preds.append(pred)
            wer_metrics= wer.compute(
                predictions=preds,
                references=trans,)
            print(f"overall WER for {loc} with HF: {round(wer_metrics*100, 2)}")
            with open(out_file, "w") as json_file:
                for tran, pred in zip(trans, preds):
                    out_dict = {"tran": tran, "pred": pred}
                    json.dump(out_dict, json_file)
                    json_file.write("\n")


def get_longform_openai(info_file):
    wer = evaluate.load("wer")
    preds = []
    trans = []
    loc = os.path.splitext(os.path.basename(info_file))[0]
    normalizer = BasicTextNormalizer()
    out_file = f"../whisper-output/{loc}-openai.json"
    if os.path.exists(out_file):
        pass
    else:
        model = whisper.load_model("large-v2")
        with open(info_file, "r") as json_file:
            info_obj = list(json.load(json_file).values())
            wav_files = [item['wav'] for item in info_obj]
            text_files = [item['txt'] for item in info_obj]
            for wav, text in zip(wav_files, text_files):
                tran_df = pd.read_csv(text, sep="\t")
                # NOTE: it indeed includes the talking clips from interviewees
                # tran_df = tran_df[~tran_df['Spkr'].str.contains(r'(int|misc)', case=False)]
                tran_df['clean_tran'] = tran_df["Content"].apply(clean_txt)
                tran_df.replace('', pd.NA, inplace=True)
                tran_df.dropna(inplace=True)
                text = ". ".join(tran_df["clean_tran"].values.tolist())
                text = normalizer(text)
                pred = model.transcribe(wav)
                pred = pred['text']
                pred = clean_output(pred)
                pred = normalizer(pred)
                trans.append(text)
                preds.append(pred)
            wer_metrics= wer.compute(
                predictions=preds,
                references=trans,)
            print(f"overall WER for {loc} with OpenAI: {round(wer_metrics*100, 2)}")
            with open(out_file, "w") as json_file:
                for tran, pred in zip(trans, preds):
                    out_dict = {"tran": tran, "pred": pred}
                    json.dump(out_dict, json_file)
                    json_file.write("\n")


def get_metrics(file_list, framework):
    trans = []
    preds = []
    wer = evaluate.load("wer")
    for file in file_list:
        with open(file, "r") as json_file:
            for line in json_file:
                data = json.loads(line)
                trans.append(data['tran'])
                preds.append(data['pred'])
    trans = [re.sub(r'\b(?:huh|uh)\b', "", item) for item in trans]
    trans = [re.sub(r"\s+", " ", item) for item in trans]
    preds = [re.sub(r"\s+", " ", item) for item in preds]
    wer_metrics= wer.compute(
        predictions=preds,
        references=trans,)
    print(f"CORAAL overall WER with {framework}: {round(wer_metrics*100, 2)}")


if __name__ == "__main__":
    start_time = datetime.now()
    config = configparser.ConfigParser()
    config.read("config.ini")
    # CORAAL 2021 version
    files = glob("../coraal-files/*.json")
    if len(files) == 0:
        get_files(config['DATA']['coraal-2021'])
    else:
        if len(glob(f"../whisper-output/*.json")) == 0:
            for each_file in files:
                get_longform_hf(each_file)
                get_longform_openai(each_file)
        else:
            files = glob("../whisper-output/*.json")
            hf_files = [item for item in files if "-hf" in item]
            openai_files = [item for item in files if "-openai" in item]
            get_metrics(hf_files, "huggingface")
            get_metrics(openai_files, "openai")
    print(f"Total running time: {datetime.now() - start_time}")
