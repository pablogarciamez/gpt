import torch
import torch.nn as nn
import torch.nn.functional as F

class BigramLM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        nn.init.normal_(self.token_embedding_table.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))

        return logits, loss

if __name__ == "__main__":
    from data import V, train_data, get_batch
    import math

    torch.manual_seed(1337)
    model = BigramLM(V)
    x, y = get_batch(train_data, 4, 8)
    logits, loss = model(x, y)
    assert logits.shape == (4, 8, 65)
    print(loss.item())
    print(math.log(V))