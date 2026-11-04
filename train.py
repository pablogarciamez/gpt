import torch
import math
import torch.nn.functional as F
from data import V, get_batch, train_data, val_data, decode
from models.bigram import BigramLM

batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
eval_iters = 200
learning_rate = 1e-2
device = "cuda" if torch.cuda.is_available() else "cpu"

@torch.no_grad()
def estimate_loss(model, splits, eval_iters, batch_size, block_size, device):
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(data, batch_size, block_size)
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out

@torch.no_grad()
def generate(model, idx, max_new_tokens, block_size):
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    model.train()
    return idx

def main():
    torch.manual_seed(1337)
    model = BigramLM(V).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    idx = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(generate(model, idx, 300, block_size)[0].tolist()))

    splits = {"train": train_data, "val": val_data}
    for it in range(max_iters):
        if it % eval_interval == 0 or it == max_iters - 1:
            losses = estimate_loss(model, splits, eval_iters, batch_size, block_size, device)
            bpc = losses["val"] / math.log(2)
            print(f"it: {it}, train_loss: {losses['train']:.4f}, val_loss: {losses['val']:.4f}, bpc: {bpc:.4f}")
        x, y = get_batch(train_data, batch_size, block_size)
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print(decode(generate(model, idx, 300, block_size)[0].tolist()))

if __name__ == "__main__":
    main()