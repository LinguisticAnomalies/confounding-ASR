'''
WER error analysis
'''
import re
from datetime import datetime
from glob import glob
import warnings
import pandas as pd
import evaluate
warnings.filterwarnings('ignore')



def get_val_stat(model_type, epoch=2):
    files = glob("../whisper-output-clean/val/*.csv")
    if model_type == "ft":
        files = [item for item in files if "ft" in item]
        if epoch == 4:
            files = [item for item in files if "4" in item]
        else:
            files = [item for item in files if "4" not in item]
    else:
        files = [item for item in files if "ft" not in item]
    
    locs = ("ATL", "DCA", "DCB", "LES", "PRV", "ROC", "VLD")
    print("Validation set")
    for loc in locs:
        loc_files = [item for item in files if loc in item]
        for input_file in loc_files:
            model_name = input_file.split("_")[0].split("/")[-1]
            loc = input_file.split("_")[-2]
            print(f"{model_type} {model_name} on {loc}")
            pred_df = pd.read_csv(input_file)
            pred_df['prediction'] = pred_df["prediction"].str.lower()
            pred_df['prediction'] = pred_df["prediction"].apply(
                lambda x: re.sub(r'[^a-zA-Z0-9\s\']+$', '', str(x)))
            pred_df['prediction'] = pred_df["prediction"].str.strip()
            overall_wer = evaluate.load("wer")
            overall_cer = evaluate.load("cer")
            wer= overall_wer.compute(
                predictions=pred_df['prediction'].values.tolist(),
                references=pred_df['transcription'].values.tolist()
            )
            cer = overall_cer.compute(
                predictions=pred_df['prediction'].values.tolist(),
                references=pred_df['transcription'].values.tolist()
            )
            print(f"overall WER: {round(wer*100, 2)}, CER:{round(cer*100, 2)}")
            print("==============")


def get_test_stat():
    print("Pre-trained on the test set")
    overall_wer = evaluate.load("wer")
    overall_cer = evaluate.load("cer")
    pred_df = pd.read_csv(
        "../whisper-output-clean/whisper-large-v2.csv")
    pred_df['prediction'] = pred_df["prediction"].str.lower()
    pred_df['prediction'] = pred_df["prediction"].apply(
        lambda x: re.sub(r'[^a-zA-Z0-9\s\']+$', '', str(x)))
    pred_df['prediction'] = pred_df["prediction"].str.strip()
    wer= overall_wer.compute(
        predictions=pred_df['prediction'].values.tolist(),
        references=pred_df['transcription'].values.tolist()
    )
    cer = overall_cer.compute(
        predictions=pred_df['prediction'].values.tolist(),
        references=pred_df['transcription'].values.tolist()
    )
    print(f"overall WER: {round(wer*100, 2)}, CER:{round(cer*100, 2)}")
    print("Fine-tuned on the test test")
    pred_df = pd.read_csv(
        "../whisper-output-clean/whisper-large-v2_ft_4.csv")
    pred_df['prediction'] = pred_df["prediction"].str.lower()
    pred_df['prediction'] = pred_df["prediction"].apply(
        lambda x: re.sub(r'[^a-zA-Z0-9\s\']+$', '', str(x)))
    pred_df['prediction'] = pred_df["prediction"].str.strip()
    wer= overall_wer.compute(
        predictions=pred_df['prediction'].values.tolist(),
        references=pred_df['transcription'].values.tolist()
    )
    cer = overall_cer.compute(
        predictions=pred_df['prediction'].values.tolist(),
        references=pred_df['transcription'].values.tolist()
    )
    print(f"overall WER: {round(wer*100, 2)}, CER:{round(cer*100, 2)}")


def process_files_for_age(files, locs):
    ag_map = {
        "ATL": {
            "1": "under 29",
            "2": "30-50"
        },
        "DCA": {
            "1": "under 19",
            "2": "20-29",
            "3": "30-50",
            "4": "> 51"
        },
        "DCB": {
            "1": "under 19",
            "2": "20-29",
            "3": "30-50",
            "4": "> 51"
        },
        "LES": {
            "2": "20-29",
            "3": "30-50",
            "4": "> 51"
        },
        "PRV": {
            "1": "under 29",
            "2": "30-50",
            "3": "> 51"
        },
        "ROC": {
            "1": "under 29",
            "2": "30-50",
            "3": "> 51"
        },
        "VLD": {
            "2": "under 29",
            "3": "30-50",
            "4": "> 51"
        },
    }
    for loc in locs:
        loc_files = [item for item in files if loc in item]
        for input_file in loc_files:
            loc = input_file.split("_")[-2]
            epoch = input_file.split("_")[-1].split(".")[0]
            if "ft" in input_file:
                print(f"Fine-tuned whisper on {loc}, epoch: {epoch}")
            else:
                print(f"Pre-trained whisper on {loc}, epoch: {epoch}")
            pred_df = pd.read_csv(input_file)
            pred_df['prediction'] = pred_df["prediction"].str.lower()
            pred_df['prediction'] = pred_df["prediction"].apply(
                lambda x: re.sub(r'[^a-zA-Z0-9\s\']+$', '', str(x)))
            pred_df['prediction'] = pred_df["prediction"].str.strip()
            # if loc == "DCA" or loc == "DCB":
            #     get_utt_stats_se(pred_df, loc, "path", r'se(\d+)', "se")
            # else:
            #     pass
            get_utt_stats(pred_df, ag_map, loc, "path", r'ag(\d+)', "ag")
            print("==============")

def get_utt_stats_se(input_df, loc, col, col_pattern, new_col):
    input_df[new_col] = input_df[col].str.extract(col_pattern)
    ags = input_df[new_col].unique()
    for ag in ags:
        print(f"SE group: {ag} in {loc}")
        sub_df = input_df.loc[input_df[new_col] == ag]
        overall_wer = evaluate.load("wer")
        overall_cer = evaluate.load("cer")
        wer = overall_wer.compute(
            predictions=sub_df['prediction'].values.tolist(),
            references=sub_df['transcription'].values.tolist()
        )
        cer = overall_cer.compute(
            predictions=sub_df['prediction'].values.tolist(),
            references=sub_df['transcription'].values.tolist()
        )
        result = {
                'ag': ag,
                'wer': round(wer * 100, 2),
                'cer': round(cer * 100, 2)
            }
        print(f"overall WER: {result['wer']}, CER: {result['cer']}")
        print("---------")


def get_utt_stats(input_df, ag_map, loc, col, col_pattern, new_col):
    input_df[new_col] = input_df[col].str.extract(col_pattern)
    ags = input_df[new_col].unique()
    # special case for DC
    if  loc == "DCB" or loc =="DCA":
        ags = ["2", "3", "4"]
        for ag in ags:
            if ag == "2":
                print("Age group under 29")
                sub_df = input_df.loc[input_df['ag'].isin(["1", "2"])]
            else:
                print(f"Age group {ag_map[loc][ag]}")
                sub_df = input_df.loc[input_df['ag'] == ag]
            overall_wer = evaluate.load("wer")
            overall_cer = evaluate.load("cer")
            wer = overall_wer.compute(
                predictions=sub_df['prediction'].values.tolist(),
                references=sub_df['transcription'].values.tolist()
            )
            cer = overall_cer.compute(
                predictions=sub_df['prediction'].values.tolist(),
                references=sub_df['transcription'].values.tolist()
            )
            result = {
                'ag': ag,
                'wer': round(wer * 100, 2),
                'cer': round(cer * 100, 2)
            }
            print(f"overall WER: {result['wer']}, CER: {result['cer']}")
            print("---------")
    else:
        for ag in ags:
            print(f"Age group {ag_map[loc][ag]}")
            sub_df = input_df.loc[input_df['ag'] == ag]
            overall_wer = evaluate.load("wer")
            overall_cer = evaluate.load("cer")
            wer = overall_wer.compute(
                predictions=sub_df['prediction'].values.tolist(),
                references=sub_df['transcription'].values.tolist()
            )
            cer = overall_cer.compute(
                predictions=sub_df['prediction'].values.tolist(),
                references=sub_df['transcription'].values.tolist()
            )
            result = {
                'ag': ag,
                'wer': round(wer * 100, 2),
                'cer': round(cer * 100, 2)
            }
            print(f"overall WER: {result['wer']}, CER: {result['cer']}")
            print("---------")

if __name__ == "__main__":
    start_time = datetime.now()
    # get_val_stat("ft", 4)
    # get_test_stat()
    files = glob("../whisper-output-clean/val/*.csv")
    files = [item for item in files if "ft" in item]
    files = [item for item in files if "4" in item]
    locs = ("ATL", "DCA", "DCB", "LES", "PRV", "ROC", "VLD")
    process_files_for_age(files, locs)
    print(f"Total running time: {datetime.now() - start_time}")

