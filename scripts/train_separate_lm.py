'''
Prepare a component from CORAAL for character-level language modeling.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the encoder and decoder and some other related info.
Adapted from nanoGPT GitHub repo
'''


from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import re
import os
import json
import configparser
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from scipy import stats as st
device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
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
    txtout = re.sub(r'\b(?:huh|uh)\b', "", txtout)
    # ooh -> oh
    txtout = re.sub(r"(?i)ooh", "oh", txtout)
    txtout = re.sub(r'\([^)]*\)', "", txtout)
    txtout = re.sub(r"\/[^)]*\/", "", txtout)
    txtout = re.sub(r"\<[^)]*\>", "", txtout)
    txtout = re.sub(r'[^a-zA-Z0-9\s\'-]', "", txtout)
    # non-breaking spaces
    txtout = txtout.replace("\xa0", " ")
    return txtout.lower().strip()


def get_data(loc):
    info_file = f"../coraal-files/{loc}.json"
    total_trans = []
    with open(info_file, "r") as json_file:
        info_obj = list(json.load(json_file).values())
        text_files = [item['txt'] for item in info_obj]
        for text_file in text_files:
            tran_df = pd.read_csv(text_file, sep="\t")
            tran_df = tran_df[~tran_df['Spkr'].str.contains(r'(int|misc)', case=False)]
            tran_df['clean_tran'] = tran_df["Content"].apply(clean_txt)
            tran_df['clean_tran'] = tran_df['clean_tran'].str.replace("-", '')
            tran_df.replace('', pd.NA, inplace=True)
            tran_df.dropna(inplace=True)
            trans = tran_df['clean_tran'].values.tolist()
            total_trans.extend(trans)
    return total_trans

def get_batch_for_batch(split):
    # generate a batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
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
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


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


def calculate_perplexity(model, sentence):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        sentence = torch.tensor(sentence, dtype=torch.long, device=device).unsqueeze(0)
        logits, _ = model(sentence)
        logits = logits.squeeze(0)  # Remove the batch dimension
        probs = F.softmax(logits, dim=-1)
        targets = sentence[:, 1:]  # Assuming the next token is the target
        targets = targets.contiguous().view(-1)  # Flatten targets
        loss = F.cross_entropy(logits[:-1].contiguous().view(-1, logits.size(-1)), targets)
        perplexity = torch.exp(loss)
    return perplexity.item()

def ppl_driver(model, eval_list):
    ppls = []
    for sent in tqdm(eval_list, total=len(eval_list), desc="calculating PPLs"):
        sent = torch.tensor(encode(sent), dtype=torch.long)
        ppl = calculate_perplexity(model, sent)
        ppls.append(ppl)
    ppls = [x for x in ppls if not np.isnan(x)]
    print(f"PPL length: {len(ppls)}")
    intervals = st.t.interval(
        confidence=0.95, df=len(ppls)-1,
        loc=np.mean(ppls), scale=st.sem(ppls))
    return np.mean(ppls), intervals

def get_val_ppl(model, eval_list):
    ppls = []
    for sent in tqdm(eval_list, total=len(eval_list), desc="calculating PPLs"):
        sent = torch.tensor(encode(sent), dtype=torch.long)
        ppl = calculate_perplexity(model, sent)
        ppls.append(ppl)
    return ppls

def get_val_ppl_driver(locs):
    total_ppls = pd.DataFrame()
    for loc in locs:
        print(f"Evaluating GPT on {loc}")
        model_name = f"../fine-tuned/gpt-{loc}.pth"
        val_df = pd.read_csv(
            os.path.join(config_parser['DATA']['val'], loc, "metadata.csv"))
        model = GPTLanguageModel(
                vocab_size, n_embd, block_size, n_head, n_layer)
        model.load_state_dict(torch.load(model_name))
        model.to(device)
        val_sents = val_df['transcription'].values.tolist()
        val_ppls = get_val_ppl(model, val_sents)
        val_df['ppl'] = val_ppls
        total_ppls = pd.concat([total_ppls, val_df])
    total_ppls.to_csv("../val_ppls.csv", index=False)
        


if __name__ == "__main__":
    start_time = datetime.now()
    torch.manual_seed(42)
    config_parser = configparser.ConfigParser()
    config_parser.read("config.ini")
    locs = ("ATL", "DCA", "DCB", "LES", "PRV", "ROC", "VLD")
    characters = [chr(x) for x in range(ord('a'), ord('z') + 1)]
    numbers = [str(x) for x in range(10)]
    chars = characters + numbers + [" ", "'"]
    vocab_size = len(chars)
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    ci_data = []
    for train_component in locs:
        model_name = f"../fine-tuned/gpt-{train_component}.pth"
        text = get_data(train_component)
        text = " ".join(text)
        stoi = { ch:i for i,ch in enumerate(chars) }
        itos = { i:ch for i,ch in enumerate(chars) }
        # treat oov as space
        # encode = lambda s: [stoi.get(c, 0) for c in s]
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l])
        if os.path.exists(model_name):
            pass
            # print(f"Evaluating GPT on {train_component}")
            # model = GPTLanguageModel(
            #     vocab_size, n_embd, block_size, n_head, n_layer)
            # model.load_state_dict(torch.load(model_name))
            # model.to(device)
            # for val_component in locs:
            #     ppls = []
            #     if val_component != train_component:
            #         val_text = get_data(val_component)
            #         ppl, interval = ppl_driver(model, val_text)
            #         record = {
            #             "train_corpus": train_component,
            #             "eval_corpus": val_component,
            #             "ppl": ppl,
            #             "lower_ci": interval[0],
            #             "upper_ci": interval[1]}
            #         ci_data.append(record)
            #         print(f"GPT-{train_component} ppl on {val_component}: {ppl}, 95% lower CI: {interval[0]}, 95% upper CI: {interval[1]}")
            #         print("------------------")
            # ci_df = pd.DataFrame(ci_data)
            # ci_df.to_csv("../char_gpt_ppl_coraal.csv", index=False)
        else:
            # training and validation split
            print(f"Training GPT on {train_component}")
            text = get_data(train_component)
            text = " ".join(text)
            data = torch.tensor(encode(text), dtype=torch.long)
            n = int(0.9*len(data))
            train_data = data[:n]
            val_data = data[n:]
            print(f"traing set length: {len(train_data)}")
            print(f"val set length: {len(val_data)}")
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
                    print("====================")
                    model.eval()
                    val_loss = 0.0
                    losses = estimate_loss_iter()
                    print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                    if losses['val'] < best_val_loss:
                        best_val_loss = losses['val']
                        best_model_state_dict = model.state_dict()
                    else:
                        counter += 1
                        if counter > patience:
                            print("early stopping")
                            break
            torch.save(best_model_state_dict, model_name)
    get_val_ppl_driver(locs)
    print(f"Total running time: {datetime.now() - start_time}")

