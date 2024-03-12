import configparser
from glob import glob
from TRESTLE import (
    TextWrapperProcessor,
    AudioProcessor
)


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("config.ini")
    cha_txt_patterns = {
        r'\([^)]*\)': "",
        r'(\w)\1\1': '',
        r'\[.*?\]': "",
        r'&-(\w+)': r'\1',
        r'&+(\w+)': r'\1',
        r'<(\w+)>': r'\1',
        r'\+...': "",
        r'[^A-Za-z\n \']': '',
        r'\s+': ' ',
    }
    wls_sample = {
        "format": ".cha",
        "text_input_path": config['DATA']['wls_text_input'],
        "audio_input_path": config['DATA']['wls_audio_input'],
        "text_output_path": config['DATA']['wls_text_output'],
        "audio_output_path": config['DATA']['wls_audio_output'],
        "audio_type": ".mp3",
        "speaker": "*PAR",
        "content": r'@Bg:	Activity\n.*?@Eg:	Activity',
    }
    # if len(glob(f"{config['DATA']['wls_text_output']}/*.jsonl")) > 0:
    #     if len(glob(f"{config['DATA']['wls_audio_output']}/*.wav")) > 0:
    #         pass
    #     else:
    #         processor = AudioProcessor(data_loc=wls_sample, sample_rate=16000)
    #         processor.process_audio()
    # else:
    #     wrapper_processor = TextWrapperProcessor(
    #         data_loc=wls_sample, txt_patterns=cha_txt_patterns)
    #     wrapper_processor.process()
    #     processor = AudioProcessor(data_loc=wls_sample, sample_rate=16000)
    #     processor.process_audio()
    wrapper_processor = TextWrapperProcessor(
        data_loc=wls_sample, txt_patterns=cha_txt_patterns)
    wrapper_processor.process()