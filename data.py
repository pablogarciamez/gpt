import torch
from pathlib import Path

def encode(s):
    return [stoi[c] for c in s]

def decode(ids):
    return ''.join([itos[i] for i in ids])

def get_batch(data, batch_size, block_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i+block_size] for i in ix])
    y = torch.stack([data[i+1 : i+block_size+1] for i in ix])
    return x, y


path = Path(__file__).parent / "data" / "input.txt"
with open(path, encoding="utf-8") as inputTxt:
    text = inputTxt.read()
    chars = sorted(set(text))
    V = len(chars)

    stoi = {char: i for i, char in enumerate(chars)}
    itos = {i: char for i, char in enumerate(chars)}

    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

if __name__ == "__main__":
    assert decode(encode(text)) == text
    assert V == 65
    assert data.shape == torch.Size([1115394])
    assert data.dtype == torch.int64
    assert len(train_data) + len(val_data) == len(data)

    torch.manual_seed(1337)
    x, y = get_batch(train_data, 4, 8)
    assert x.shape == torch.Size([4, 8])
    assert y.shape == torch.Size([4, 8])
    assert torch.equal(x[:, 1:], y[:, :-1])
