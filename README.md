## Confounding ASR

This repository contains code for the manuscript *Examining Racial Disparities in Automatic Speech Recognition Performance: Potential Confounding by Provenance*.

## Setup

Our code is built with CUDA 12.1 with conda and Python 3.10 via the following code:

```shell
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

Please install dependency packages using `pip install -r requirements.txt` or `conda install --yes --file requirements.txt`

Please also make sure you have [FFmpeg](https://ffmpeg.org/) and [sox](https://sourceforge.net/projects/sox/) installed in your environment.

Before start, please create a `config.ini` file under the scripts folder, using the following template:

```
[DATA]
coraal-2021 = /path/to/coraal/2021/release
data = /path/to/post/processing/data/folder
train = /path/to/post/processing/data/training/split
test = /path/to/post/processing/data/test/split
val = /path/to/post/processing/data/validation/split
val_noise_profile = /path/to/post/processing/data/validation/split/for/noise/profile
val_nr = /path/to/post/processing/data/validation/split/after/noise/reduction
```

## Dataset

The [CORAAL](https://oraal.uoregon.edu/coraal) is publicly available online. You can find *This Side of The River - The Story of Princeville* on [YouTube](https://www.youtube.com/watch?v=KhRUSZoJ5_Y).

## Folders

## Folders
The structure of the repo is listed as follows:

```
├── coraal-files
├── scripts
│   ├── fine_tune_whisper.py
│   ├── build_dataset.py
│   ├── error_analysis.py
│   ├── get_stat.py
│   ├── prepare_char.py
│   ├── reduce_noise.py
│   ├── run_baseline.py
│   ├── transcribe_coraal.py
│   ├── transcribe_vali.py
├── ft-models
├── whisper-output-clean
```

The fine-tuned models are saved under the `ft-models` folders, and the generated transcripts are saved under `whisper-output-clean` folder. The `coraal-files` contains the files for the baseline comparison.

## To reproduce

### Preprocessing

We use [TRESTLE](https://github.com/LinguisticAnomalies/harmonized-toolkit) to preprocessing the text and audio utterances. Please refer to TRESTLE's GitHub repository for more information.

### Scripts usage


To start, please first run `run_baseline.py` to get necessary files and results for the baseline model.


To build the *clean* training/test/validation split used in this study, one can run `build_dataset.py`. This script follows a similar strategy as TRESTLE, which performs basic text preprocessing as defined by the users and resampled the audio utterances into 16k Hz.

One can run `fine_tune_whisper.py` for fine-tuning whisper-large-v2 model. The pre-trained model is accessed from [HuggingFace](https://huggingface.co/openai/whisper-large-v2).

`transcribe_coraal.py` and `transcribe_vali.py` are designed to use either pre-trained or fine-tuned whisper-large-v2 to transcribe the test and validation split, respectively. The transcriptions are saved under `whisper-output-clean` folder.

We use `prepare_char.py` to train the character-level GPT models on each CORAAL component. The trained models (*.pth) are saved under `ft-models`.


We use `error_analysis.py` and `get_stat.py` to perform the error analysis and ground truth statistics for CORAAL.


