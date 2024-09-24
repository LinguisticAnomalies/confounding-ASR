'''
Evaluate MOS on CORAAL
'''

import os
import configparser
import warnings
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from mos_wav2vec2 import MosPredictor
from transformers import set_seed
warnings.filterwarnings('ignore')


class CoraalDataset(Dataset):
    def __init__(self, data_dir, placeholder_score=3.0):
        self.data_dir = data_dir
        self.metadata_path = os.path.join(data_dir, 'metadata.csv')
        self.metadata = pd.read_csv(self.metadata_path)
        self.placeholder_score = placeholder_score
        self.valid_files = []
        self.skipped_files = []
        self.validate_audio_files()
        self.update_metadata_and_files()
        
    def validate_audio_files(self):
        for _, row in tqdm(self.metadata.iterrows(), total=len(self.metadata), desc="Validating audio files"):
            wav_path = os.path.join(self.data_dir, row['file_name'])
            try:
                torchaudio.info(wav_path)
                self.valid_files.append(row)
            except Exception as e:
                self.skipped_files.append((row['file_name'], str(e)))
        
        print(f"Total files: {len(self.metadata)}")
        print(f"Valid files: {len(self.valid_files)}")
        print(f"Skipped files: {len(self.skipped_files)}")

    def update_metadata_and_files(self):
        # Update metadata
        valid_filenames = [row['file_name'] for row in self.valid_files]
        self.metadata = self.metadata[self.metadata['file_name'].isin(valid_filenames)]
        
        # Save updated metadata
        self.metadata.to_csv(self.metadata_path, index=False)
        print(f"Updated metadata saved to {self.metadata_path}")

        # Delete skipped files
        for file_name, error in self.skipped_files:
            file_path = os.path.join(self.data_dir, file_name)
            try:
                os.remove(file_path)
                print(f"Deleted skipped file: {file_name}")
            except Exception as e:
                print(f"Error deleting {file_name}: {str(e)}")

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        wav_path = os.path.join(self.data_dir, row['file_name'])
        wav, _ = torchaudio.load(wav_path)
        if wav.shape[0] > 1:  # If stereo, convert to mono
            wav = wav.mean(dim=0, keepdim=True)
        return wav, self.placeholder_score, row['file_name']

    def __len__(self):
        return len(self.metadata)

    @staticmethod
    def collate_fn(batch):
        """
        Collate function for DataLoader.
        Args:
            batch (list): List of items from __getitem__.
        Returns:
            tuple: (wavs, scores, wavnames) where wavs is a tensor of audio data,
                   scores is a tensor of placeholder MOS scores, and wavnames is a list of filenames.
        """
        wavs, scores, wavnames = zip(*batch)
        max_len = max(wav.shape[1] for wav in wavs)
        output_wavs = torch.stack([F.pad(wav, (0, max_len - wav.shape[1])) for wav in wavs])
        output_scores = torch.tensor(scores, dtype=torch.float)
        return output_wavs, output_scores, wavnames


def load_coraal_split(root_dir: str, batch_size: int, shuffle: bool = False) -> tuple:
    """
    Load a split of the CORAAL dataset.

    Args:
        root_dir (str): Root directory of the dataset.
        batch_size (int): Batch size for the DataLoader.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to False.

    Returns:
        tuple: (dataset, dataloader) where dataset is a CoraalDataset instance
               and dataloader is a DataLoader instance.
    """
    dataset = CoraalDataset(root_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=CoraalDataset.collate_fn)
    return dataset, dataloader


@torch.no_grad()
def get_prediction(split, config):
    """
    Get MOS predictions for a specific split of the dataset.

    Args:
        split (str): The split to process ('train', 'val', or 'test').
        config (ConfigParser): Configuration object containing dataset paths.

    Returns:
        list: List of dictionaries containing predictions for each audio file.
    """
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    model = MosPredictor.from_pretrained("../fine-tuned/mos-wav2vec2_cosine").to(device)
    model.eval()

    predictions = []
    base_data_dir = config['DATA'][split]

    def process_batch(batch, split):
        wavs, _, wavnames = batch
        wavs = wavs.to(device)
        outputs = model(wavs)
        return [
            {"loc": name[:3], "MOS": float(output), "split": split, "path": name}
            for name, output in zip(wavnames, outputs.cpu().numpy())
        ]

    if split == "val":
        locs = ("ATL", "DCA", "DCB", "LES", "PRV", "ROC", "VLD")
        for loc in locs:
            data_dir = os.path.join(base_data_dir, loc)
            predictions.extend(process_location(data_dir, split, process_batch))
    else:
        predictions.extend(process_location(base_data_dir, split, process_batch))

    return predictions

def process_location(data_dir, split, process_batch):
    dataset = CoraalDataset(data_dir)
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=CoraalDataset.collate_fn,
        num_workers=4
    )
    return [
        pred
        for batch in tqdm(loader, desc=f"Predicting on {split} split")
        for pred in process_batch(batch, split)
    ]


if __name__ == "__main__":
    set_seed(42)
    all_predictions = []
    config_parser = configparser.ConfigParser()
    config_parser.read("config.ini")
    split_predictions = get_prediction("val", config_parser)
    # if os.path.exists("../mos_pred_mos.csv"):
    #     coraal_mos = pd.read_csv("../mos_pred_mos.csv")
    # else:
    #     splits = ["train", "test", "val"]
    #     for split in splits:
    #         split_predictions = get_prediction(split, config_parser)
    #         all_predictions.extend(split_predictions)
    #     pred_df = pd.DataFrame(all_predictions)
    #     pred_df.to_csv("../mos_pred_mos.csv", index=False)
    # if os.path.exists("../mos_pred_river.csv"):
    #     river_mos = pd.read_csv("../mos_pred_river.csv")
    # else:
    #     river_pred = get_prediction('river', config_parser)
    #     all_predictions.extend(river_pred)
    #     pred_df = pd.DataFrame(all_predictions)
    #     pred_df['loc'] = "river"
    #     pred_df.to_csv("../mos_pred_river.csv", index=False)
    # total_df = pd.concat([coraal_mos, river_mos])
    # total_df.to_csv("../mos_total.csv", index=False)
   