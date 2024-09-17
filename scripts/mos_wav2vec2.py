'''
Adapt MOS on pre-trained Wav2Vec2 base model
Adapted from: https://github.com/nii-yamagishilab/mos-finetune-ssl
'''
import argparse
import logging
from tqdm import tqdm
from datetime import datetime
import os
import json
import warnings
import configparser
from dataclasses import dataclass
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import set_seed, Wav2Vec2Model
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import CosineAnnealingLR

warnings.filterwarnings('ignore')

@dataclass
class TrainingConfig:
    """
    Configuration class for training parameters.

    Attributes:
        model (nn.Module): The neural network model to be trained.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        criterion (nn.Module): Loss function for training.
        optimizer (optim.Optimizer): Optimizer for updating model parameters.
        scheduler (optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        device (torch.device): Device to run the training on (CPU or GPU).
        num_epochs (int): Number of training epochs.
        patience (int): Number of epochs to wait before early stopping.
        save_dir (str): Directory to save the trained model.
        scheduler_type (str): Type of scheduler to use ('cosine' or 'plateau').
    """
    model: nn.Module
    train_loader: DataLoader
    val_loader: DataLoader
    criterion: nn.Module
    optimizer: optim.Optimizer
    scheduler: optim.lr_scheduler._LRScheduler
    device: torch.device
    num_epochs: int
    patience: int
    save_dir: str
    scheduler_type: str


def setup_logging(log_dir: str, scheduler_type: str) -> str:
    """
    Set up logging configuration with unique filenames.

    Args:
        log_dir (str): Directory to store log files.
        scheduler_type (str): Type of scheduler being used.

    Returns:
        str: Path to the created log file.
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_filename = f"training_{scheduler_type}.log"
    log_filepath = os.path.join(log_dir, log_filename)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] - %(message)s',
        handlers=[
            logging.FileHandler(log_filepath),
            logging.StreamHandler()
        ]
    )
    
    return log_filepath

class VoiceMOSDataset(Dataset):
    """
    Dataset class for Voice MOS (Mean Opinion Score) data.

    Args:
        root_dir (str): Root directory of the dataset.
        split (str): Dataset split ('train', 'validation', or 'test').
    """

    def __init__(self, root_dir: str, split: str):
        self.root_dir = root_dir
        self.split = split
        self.data_dir = os.path.join(root_dir, split)
        self.metadata_path = os.path.join(self.data_dir, 'metadata.csv')
        self.df = pd.read_csv(self.metadata_path)
        self.valid_indices = [idx for idx, row in self.df.iterrows() if os.path.exists(os.path.join(self.data_dir, row['file_name']))]

    def __getitem__(self, idx: int):
        """
        Get a single item from the dataset.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            tuple: (wav, score, wavname) where wav is the audio tensor,
                   score is the MOS score, and wavname is the filename.
        """
        real_idx = self.valid_indices[idx]
        row = self.df.iloc[real_idx]
        wavname = row['file_name']
        score = float(row['label'])
        wavpath = os.path.join(self.data_dir, wavname)
        wav, _ = torchaudio.load(wavpath)
        return wav, score, wavname
    
    def __len__(self):
        """
        Get the number of items in the dataset.

        Returns:
            int: Number of items in the dataset.
        """
        return len(self.valid_indices)
    
    @staticmethod
    def collate_fn(batch):
        """
        Collate function for DataLoader.

        Args:
            batch (list): List of items from __getitem__.

        Returns:
            tuple: (wavs, scores, wavnames) where wavs is a tensor of audio data,
                   scores is a tensor of MOS scores, and wavnames is a list of filenames.
        """
        wavs, scores, wavnames = zip(*batch)
        max_len = max(wav.shape[1] for wav in wavs)
        output_wavs = torch.stack([F.pad(wav, (0, max_len - wav.shape[1])) for wav in wavs])
        scores = torch.tensor(scores)
        return output_wavs, scores, wavnames

class MosPredictor(nn.Module):
    """
    Neural network model for predicting MOS scores.

    Args:
        wav2vec2_model (nn.Module): Pre-trained Wav2Vec2 model.
    """

    def __init__(self, wav2vec2_model: nn.Module):
        super(MosPredictor, self).__init__()
        self.wav2vec2 = wav2vec2_model
        self.output_layer = nn.Linear(self.wav2vec2.config.hidden_size, 1)
    
    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            wav (torch.Tensor): Input audio tensor.

        Returns:
            torch.Tensor: Predicted MOS scores.
        """
        wav = wav.squeeze(1)
        with torch.no_grad():
            outputs = self.wav2vec2(wav, output_hidden_states=True)
        last_hidden_state = outputs.last_hidden_state
        x = torch.mean(last_hidden_state, dim=1)
        return self.output_layer(x).squeeze(1)
    
    def save_pretrained(self, save_directory: str) -> None:
        """
        Save the model to a directory.

        Args:
            save_directory (str): Directory to save the model.
        """
        os.makedirs(save_directory, exist_ok=True)
        self.wav2vec2.save_pretrained(os.path.join(save_directory, "wav2vec2"))
        torch.save(
            self.output_layer.state_dict(),
            os.path.join(save_directory, "output_layer.pt"))
        config = {
            "wav2vec2_config": self.wav2vec2.config.to_dict(),
            "output_layer_size": self.output_layer.weight.size(1)
        }
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(config, f)
    
    @classmethod
    def from_pretrained(cls, load_directory: str) -> 'MosPredictor':
        """
        Load a pre-trained model from a directory.

        Args:
            load_directory (str): Directory containing the pre-trained model.

        Returns:
            MosPredictor: Loaded model instance.
        """
        with open(os.path.join(load_directory, "config.json"), "r") as f:
            config = json.load(f)
        wav2vec2_model = Wav2Vec2Model.from_pretrained(os.path.join(load_directory, "wav2vec2"))
        model = cls(wav2vec2_model)
        model.output_layer.load_state_dict(torch.load(os.path.join(load_directory, "output_layer.pt")))
        return model

def load_voicemos_split(root_dir: str, split: str, batch_size: int, shuffle: bool = False) -> tuple:
    """
    Load a split of the VoiceMOS dataset.

    Args:
        root_dir (str): Root directory of the dataset.
        split (str): Dataset split ('train', 'validation', or 'test').
        batch_size (int): Batch size for the DataLoader.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to False.

    Returns:
        tuple: (dataset, dataloader) where dataset is a VoiceMOSDataset instance
               and dataloader is a DataLoader instance.
    """
    dataset = VoiceMOSDataset(root_dir, split)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=VoiceMOSDataset.collate_fn)
    return dataset, dataloader


def train(config: TrainingConfig) -> nn.Module:
    """
    Train the model using the provided configuration.

    Args:
        config (TrainingConfig): Configuration object containing training parameters.

    Returns:
        nn.Module: Trained model.
    """
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(config.num_epochs):
        config.model.train()
        train_loss = 0.0
        train_steps = 0

        for wavs, scores, _ in tqdm(config.train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}"):
            wavs, scores = wavs.to(config.device), scores.to(config.device)
            config.optimizer.zero_grad()
            outputs = config.model(wavs)
            loss = config.criterion(outputs, scores)
            loss.backward()
            config.optimizer.step()
            train_loss += loss.item()
            train_steps += 1

        avg_train_loss = train_loss / train_steps
        logging.info(f"Epoch {epoch+1}/{config.num_epochs} - Average training loss: {avg_train_loss:.4f}")

        val_loss = validate(config)
        logging.info(f"Epoch {epoch+1}/{config.num_epochs} - Validation loss: {val_loss:.4f}")

        if config.scheduler_type == 'cosine':
            config.scheduler.step()
        elif config.scheduler_type == 'plateau':
            config.scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            config.model.save_pretrained(config.save_dir)
            logging.info(f"New best model saved to {config.save_dir}")
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                logging.info(f"Early stopping triggered after {epoch+1} epochs")
                break

        current_lr = config.optimizer.param_groups[0]['lr']
        logging.info(f"Current learning rate: {current_lr:.6f}")

    return config.model

def validate(config: TrainingConfig) -> float:
    """
    Validate the model using the provided configuration.

    Args:
        config (TrainingConfig): Configuration object containing validation parameters.

    Returns:
        float: Average validation loss.
    """
    config.model.eval()
    val_loss = 0.0
    val_steps = 0

    with torch.no_grad():
        for wavs, scores, _ in config.val_loader:
            wavs, scores = wavs.to(config.device), scores.to(config.device)
            outputs = config.model(wavs)
            loss = config.criterion(outputs, scores)
            val_loss += loss.item()
            val_steps += 1

    return val_loss / val_steps

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MOS Predictor")
    parser.add_argument("--scheduler", type=str, choices=['cosine', 'plateau'], default='cosine',
                        help="Type of learning rate scheduler to use")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train")
    args = parser.parse_args()

    config_parser = configparser.ConfigParser()
    config_parser.read("config.ini")

    # Hyperparameters
    batch_size = 32
    num_epochs = 50
    initial_learning_rate = 1e-4
    patience = 10
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    set_seed(42)
    log_filepath = setup_logging("logs", args.scheduler)

    logging.info(f"Log file created at: {log_filepath}")
    logging.info("Starting training with the following parameters:")
    logging.info(f"Batch size: {batch_size}")
    logging.info(f"Number of epochs: {num_epochs}")
    logging.info(f"Initial learning rate: {initial_learning_rate}")
    logging.info(f"Patience: {patience}")
    logging.info(f"Device: {device}")
    logging.info(f"Scheduler: {args.scheduler}")

    model_name = "facebook/wav2vec2-base"
    wav2vec2_model = Wav2Vec2Model.from_pretrained(model_name)
    model = MosPredictor(wav2vec2_model).to(device)

    train_dataset, train_loader = load_voicemos_split(
        config_parser['DATA']['mos'], 'train', batch_size, shuffle=True)
    val_dataset, val_loader = load_voicemos_split(
        config_parser['DATA']['mos'], 'validation', batch_size)
    test_dataset, test_loader = load_voicemos_split(
        config_parser['DATA']['mos'], 'test', batch_size)
    
    # mean absolute error
    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=initial_learning_rate)
    if args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    else:  # 'plateau'
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    save_dir = f"../fine-tuned/mos-wav2vec2_{args.scheduler}"

    training_config = TrainingConfig(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=num_epochs,
        patience=patience,
        save_dir=save_dir,
        scheduler_type=args.scheduler
    )

    model = train(training_config)

    # Final evaluation on test set
    test_config = TrainingConfig(**training_config.__dict__)
    test_config.val_loader = test_loader
    test_loss = validate(test_config)
    logging.info(f"Final test loss: {test_loss:.4f}")
