import re
import os
import json
import configparser
import csv
import pysrt
import pydub

def time_to_milliseconds(time_obj):
    # Extract hours, minutes, seconds, and microseconds
    hours = time_obj.hour
    minutes = time_obj.minute
    seconds = time_obj.second
    microseconds = time_obj.microsecond

    # Convert each unit into milliseconds
    milliseconds = (hours * 3600 + minutes * 60 + seconds) * 1000 + microseconds / 1000

    return milliseconds


def clean_txt(txtin):
    """
    text preprocessing for utterances

    :param txtin: the utterance
    :type txtin: str
    :return: the cleaned utterance
    :rtype: str
    """
    txtout = re.sub(r"\s+", " ", txtin)
    txtout = re.sub(r'\([^)]*\)', "", txtout)
    txtout = re.sub(r"\/[^)]*\/", "", txtout)
    txtout = re.sub(r"\<[^)]*\>", "", txtout)
    txtout = re.sub(r'[^a-zA-Z0-9\s]', "", txtout)
    return txtout.lower().strip()
subs = pysrt.open("river_prv.srt")

txt_manifest = "river.jsonl"
if os.path.exists(txt_manifest):
    with open(txt_manifest, "r") as jsonl_f:
        records = [json.loads(line) for line in jsonl_f]
else:
    records = []
    for item in subs:
        sub = clean_txt(item.text)
        start_time = time_to_milliseconds(item.start.to_time())
        end_time = time_to_milliseconds(item.end.to_time())
        record = {
            "start": start_time,
            "end": end_time,
            "text": sub,
            "audio": "river.mp4"}
        records.append(record)
    with open(txt_manifest, "w") as jsonl_file:
        for item in records:
            json.dump(item, jsonl_file)
            jsonl_file.write("\n")
metadata = [['file_name', 'transcription']]
config_parser = configparser.ConfigParser()
config_parser.read("config.ini")
# slide the audio
for i, record in enumerate(records):
    audio = pydub.AudioSegment.from_file("river.mp4", "mp4")
    new_file_name = f"{i}.wav"
    new_file_path = f"{config_parser['DATA']['river']}/{i}.wav"
    if i == 0:
        start = record['start']
        end = record['end'] + 1000
    else:
        start = record['start'] + 1000
        end = record['end'] + 1000
    sliced_audio = audio[start:end]
    sliced_audio = sliced_audio.set_frame_rate(16000)
    sliced_audio.export(new_file_path, format="wav")
    metadata.append([new_file_name, record['text']])
metadata_file = f"{config_parser['DATA']['river']}/metadata.csv"
with open(metadata_file, mode="w", newline="\n") as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerows(metadata)
