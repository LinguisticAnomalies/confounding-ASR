from glob import glob
import re
import pandas as pd
wer_df = pd.DataFrame()
wer_files = glob("../whisper-output-clean/val/*.csv")
locs = ("ATL", "DCA", "DCB", "LES", "PRV", "ROC", "VLD")
for loc in locs:
    loc_files = [item for item in wer_files if loc in item]
    pt_df = pd.read_csv([item for item in loc_files if "pt" in item][0])
    ft_df = pd.read_csv([item for item in loc_files if "ft" in item][0])
    loc_df = pd.merge(pt_df, ft_df, on='path')
    loc_df = loc_df[['path', "ft_wer", "pt_wer"]]
    print(loc_df.sample(n=20))
    wer_df = pd.concat([wer_df, loc_df])
mos_df = pd.read_csv("../mos_total.csv")
mos_df = mos_df.loc[mos_df['loc'] != "river"]
mos_df = mos_df.loc[mos_df['split'] == "val"]
mos_df = mos_df.rename(columns={"split": "mos_split"})
ppl_df = pd.read_csv("../val_ppls.csv")
ppl_df = ppl_df.rename(columns={"file_name": "path"})
mos_df = mos_df.merge(ppl_df, on="path")
val_df = pd.merge(mos_df, wer_df, on='path')
print(val_df.shape)
# drop outliers
# numbers only utterances
number_pattern = r'\b\d+(?:\s+\d+)*\b'
# filling words only utterances
filler_pattern = r'\b(?:m+|uh+|u+hu+h|m+h+m+|hu+h|h+m+|u+h+u+h|u+m+)\b'
combined_pattern = f'({number_pattern}|{filler_pattern})'
def contains_match(text):
    return bool(re.search(combined_pattern, text, flags=re.IGNORECASE))
mask = ~val_df['transcription'].apply(contains_match)
val_df = val_df[mask]
# WER > 1 utterances
val_df = val_df.loc[(val_df['pt_wer'] <= 1) & (val_df['ft_wer'] <= 1)]
print(val_df.shape)
print(val_df['loc'].unique())
# val_df.to_csv("../total_val.csv", index=False)