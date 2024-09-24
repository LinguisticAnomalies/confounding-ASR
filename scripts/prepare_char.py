'''
Prepare a component from CORAAL for character-level language modeling.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the encoder and decoder and some other related info.
Adapted from nanoGPT GitHub repo
'''


from datetime import datetime
import warnings
import argparse
import os
import logging
import configparser
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from scipy import stats as st
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
block_size = 512
batch_size = 64
max_iters = 50000
eval_interval = 500
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
learning_rate = 3e-4
num_epochs = 5
patience = 2
warnings.filterwarnings('ignore')

def parge_args():
    """
    add argument parser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train", action="store_true",
        help="""Indicator if train the character-level LM"""
    )
    parser.add_argument(
        "--evaluate", action="store_true",
        help="""Indicator if get utterance-level perplexity from the validation set"""
    )
    return parser.parse_args()

def setup_logging(log_dir: str, log_file: str) -> str:
    """
    Set up logging configuration with unique filenames.

    Args:
        log_dir (str): Directory to store log files.
        log_file (str): The log file name.

    Returns:
        str: Path to the created log file.
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_filepath = os.path.join(log_dir, log_file)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] - %(message)s',
        handlers=[
            logging.FileHandler(log_filepath),
            logging.StreamHandler()
        ]
    )
    
    return log_filepath


@torch.no_grad()
def estimate_loss_batch():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch_for_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

@torch.no_grad()
def estimate_loss_iter():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch_for_iter(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self, vocab_size, n_embd, block_size, n_head, n_layer):
        super().__init__()
        # Model architecture
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # Initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


def calculate_perplexity_batch(model, sentences):
    model.eval()
    with torch.no_grad():
        max_len = max(len(s) for s in sentences)
        padded_sentences = [s + [0] * (max_len - len(s)) for s in sentences]
        batch = torch.tensor(padded_sentences, dtype=torch.long, device=device)
        logits, _ = model(batch)
        
        perplexities = []
        for i, sentence in enumerate(sentences):
            sentence_len = len(sentence)
            sentence_logits = logits[i, :sentence_len-1]
            targets = batch[i, 1:sentence_len]
            
            probs = F.softmax(sentence_logits, dim=-1)
            loss = F.cross_entropy(sentence_logits, targets)
            perplexity = torch.exp(loss)
            perplexities.append(perplexity.item())
        
        return perplexities

def ppl_driver_batch(model, eval_list, loc):
    ppls = []
    for i in tqdm(range(0, len(eval_list), batch_size), desc=f"Calculating PPLs on {loc} subset"):
        batch = eval_list[i:i+batch_size]
        encoded_batch = [encode(sent) for sent in batch]
        batch_ppls = calculate_perplexity_batch(model, encoded_batch)
        ppls.extend(batch_ppls)
    return ppls

def get_batch_for_batch(split):
    # generate a batch of data of inputs x and targets y
    data = train_data if split == 'train' else test_data
    batch_indices = torch.randint(len(data) - block_size - 1, (batch_size,))
    
    batch_x = []
    batch_y = []
    for idx in batch_indices:
        x = data[idx:idx+block_size]
        y = data[idx+1:idx+block_size+1]
        batch_x.append(x)
        batch_y.append(y)
    
    x = torch.stack(batch_x)
    y = torch.stack(batch_y)
    
    x, y = x.to(device), y.to(device)
    
    return x, y


def get_batch_for_iter(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else test_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


def get_data(loc, config):
    train_df = pd.read_csv(os.path.join(config['DATA']['train'], "metadata.csv"))
    test_df = pd.read_csv(os.path.join(config['DATA']['test'], "metadata.csv"))
    val_df = pd.read_csv(os.path.join(config['DATA']['val'], loc, "metadata.csv"))
    train_df['split'] = train_df['file_name'].str[:3]
    test_df['split'] = test_df['file_name'].str[:3]
    return train_df, test_df, val_df


if __name__ == "__main__":
    start_time = datetime.now()
    config_parser = configparser.ConfigParser()
    config_parser.read("config.ini")
    torch.manual_seed(42)
    locs = ("ATL", "DCA", "DCB", "LES", "PRV", "ROC", "VLD")
    pargs = parge_args()
    # character encoder/decoder
    characters = [chr(x) for x in range(ord('a'), ord('z') + 1)]
    numbers = [str(x) for x in range(10)]
    chars = characters + numbers + [" ", "'"]
    vocab_size = len(chars)
    # OOV
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    if ' ' not in stoi:
        stoi[' '] = len(stoi)
        itos.append(' ')
    encode = lambda s: [stoi.get(c, stoi[' ']) for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    # training process
    if pargs.train:
        for train_component in locs:
            model_name = f"../fine-tuned/gpt-{train_component}.pth"
            train_df, test_df, _ = get_data(train_component, config_parser)
            train_sents = train_df['transcription'].values.tolist()
            test_sents = test_df['transcription'].values.tolist()
            if not os.path.exists(model_name):
                # training and test split
                log_filepath = setup_logging("logs", "train_char_gpt.log")
                logging.info(f"Log file created at: {log_filepath}")
                logging.info("Starting training with the following parameters:")
                logging.info(f"Batch size: {batch_size}")
                logging.info(f"Number of epochs: {num_epochs}")
                logging.info(f"Initial learning rate: {learning_rate}")
                logging.info(f"Patience: {patience}")
                logging.info(f"Device: {device}")
                logging.info(f"========== Training GPT on {train_component} ==============")
                train_sents = " ".join(train_sents)
                train_data = torch.tensor(encode(train_sents), dtype=torch.long)
                test_data = " ".join(test_sents)
                test_data = torch.tensor(encode(test_data), dtype=torch.long)
                logging.info(f"traing set length: {len(train_data)}")
                logging.info(f"test set length: {len(test_data)}")
                model = GPTLanguageModel(
                    vocab_size, n_embd, block_size, n_head, n_layer).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                best_val_loss = float('inf')
                best_model_state_dict = None
                counter = 0
                patience = 2
                for iter in range(max_iters):
                    model.train()
                    xb, yb = get_batch_for_batch('train')
                    logits, loss = model(xb, yb)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    if iter % eval_interval == 0 or iter == max_iters - 1:
                        logging.info("--------------")
                        model.eval()
                        val_loss = 0.0
                        losses = estimate_loss_iter()
                        logging.info(f"step {iter}: train loss {losses['train']:.4f}, test loss {losses['val']:.4f}")
                        if losses['val'] < best_val_loss:
                            best_val_loss = losses['val']
                            best_model_state_dict = model.state_dict()
                        else:
                            counter += 1
                            if counter > patience:
                                logging.info("early stopping")
                                break
                torch.save(best_model_state_dict, model_name)
        logging.info(f"Total running time: {datetime.now() - start_time}")
    if pargs.evaluate:
        total_ppls = pd.DataFrame()
        for train_component in locs:
            print(f"Evaluating GPT-{train_component}")
            model_name = f"../fine-tuned/gpt-{train_component}.pth"
            model = GPTLanguageModel(
                vocab_size, n_embd, block_size, n_head, n_layer)
            model.load_state_dict(torch.load(model_name))
            model.to(device)
            for val_component in locs:
                _, _, val_df = get_data(val_component, config_parser)
                val_sents = val_df['transcription'].values.tolist()
                val_path = val_df['file_name'].values.tolist()
                print(f"validation set length: {len(val_sents)}")
                ppls = ppl_driver_batch(model, val_sents, val_component)
                ppls = [item for item in ppls if item != np.nan]
                val_locs = [val_component]*len(ppls)
                train_locs= [train_component]*len(ppls)
                sub_ppl = pd.DataFrame(
                    {'ppl': ppls,
                     'train': train_locs,
                     'val': val_locs,
                     "path": val_path
                     })
                total_ppls = pd.concat([total_ppls, sub_ppl])
        total_ppls.dropna(subset=['ppl'], inplace=True)
        total_ppls.to_csv("../total_ppl.csv", index=False)