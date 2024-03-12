import torch
import torch.nn as nn
import torch.nn.functional as F
from model import TransformerDecoder
from tokenizer import Tokenizer
import argparse

#hyperparameters
batch_size = 64
block_size = 256
n_embedding = 384
n_head = 6
n_blocks = 8
dropout = 0.2
max_iters = 3000
eval_interval = 500
eval_iters = 100
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Config():
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

def load_data(args):
    with open(args.file, 'r') as f:
        text = f.read()
    
    if args.tokenizer is not None:
        tok = Tokenizer()
        tok.load(args.tokenizer)
        vocab_size = len(tok.vocab.keys())
        
        encode = tok.encode
        decode = tok.decode
    else:
        chars = sorted(list(set(text)))

        vocab_size = len(chars)

        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}

        encode = lambda s: ([stoi[ch] for ch in s])
        decode = lambda l: "".join([itos[i] for i in l])

    data = torch.tensor(encode(text)) if args.mode == 'train' else None
    
    return data, vocab_size, encode, decode

def get_batch(data):
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x,y = x.to(device), y.to(device)
    return x, y               


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='desc')

    parser.add_argument('--mode', type=str, default='train', help='train or gen')
    parser.add_argument('--run_name', type=str, default='default', help='name of the training run')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint file')
    parser.add_argument('--file', type=str, default='tiny_shakespeare.txt', help='file to train on')
    parser.add_argument('--save_text', type=str, default='more.txt', help='file to save generated text')
    parser.add_argument('--tokenizer', type=str, default=None, help='name of the tokenizer to use')
    
    args = parser.parse_args()

    data, vocab_size, encode, decode = load_data(args)

    config = Config(block_size=block_size, n_embedding=n_embedding, n_head=n_head, n_blocks=n_blocks, dropout=dropout, vocab_size=vocab_size, device=device)

    model = TransformerDecoder(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    if args.ckpt is not None:
        ckpt = torch.load(f'{args.ckpt}.pt')
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optim'])

    if args.mode == 'train':
    
        n = int(len(data) * 0.9)
        train_data, val_data = data[:n], data[n:]

        print(sum(p.numel() for p in model.parameters()), ' parameters')

        for iter in range(max_iters):

            if iter % eval_interval == 0 or iter == max_iters - 1:
                with torch.no_grad():
                    losses = torch.zeros(2, eval_iters)
                    for k in range(eval_iters):
                        x, y = get_batch(train_data)
                        logits = model(x)
                        loss = F.cross_entropy(logits.view(batch_size*config.block_size, config.vocab_size), y.view(batch_size*config.block_size))
                        losses[0, k] = loss.item()

                        x, y = get_batch(val_data)
                        logits = model(x)
                        loss = F.cross_entropy(logits.view(batch_size*config.block_size, config.vocab_size), y.view(batch_size*config.block_size))
                        losses[1, k] = loss.item()

                print(f'iter: {iter}, train loss: {losses[0].mean():.4f}, val loss: {losses[1].mean():.4f}')


            x, y = get_batch(train_data)

            logits = model(x)
            loss = F.cross_entropy(logits.view(batch_size*config.block_size, config.vocab_size), y.view(batch_size*config.block_size))
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        
        torch.save({'model':model.state_dict(),
                    'optim':optimizer.state_dict()}, f'{args.run_name}.pt')

    elif args.mode == 'gen':
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        #print(decode(model.generate(context, max_new_tokens=100)[0].tolist()))
        open(f'{args.save_text}.txt', 'w').write(decode(model.generate(context, max_new_tokens=5000)[0].tolist()))
             
