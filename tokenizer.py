import regex as re
import pickle

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

#BPE tokenizer
class Tokenizer:
    def __init__(self, regex_pattern=None): # ".*" if we don't want to use regex
        self.merges = {}
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        self.pattern = GPT4_SPLIT_PATTERN if regex_pattern is None else regex_pattern

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256
        new_id = len(self.vocab.keys())

        text_chunks = re.findall(self.pattern, text)
        ids = [list(chunk.encode('utf-8')) for chunk in text_chunks]

        for i in range(num_merges):
            stats = {}
            for chunk in ids:
                get_stats(chunk, stats)

            best = max(stats, key=stats.get)
            ids = [merge(chunk, best, new_id) for chunk in ids]
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]

            if verbose:
                print(f'replacing {best} -> {new_id}')
            new_id += 1


    def encode(self, text):
        text_bytes = text.encode('utf-8')
        ids = list(text_bytes)
        for pair, new_idx in self.merges.items():
            ids = merge(ids, pair, new_idx)

        return ids
        

    def decode(self, ids):
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode('utf-8', errors='replace')
        return text
    
    def save(self, file_name):
        tmp = {'merges': self.merges,
               'vocab' : self.vocab}
        with open(f'{file_name}.pickle', 'wb') as f:
            pickle.dump(tmp, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, file_name):
        with open(f'{file_name}.pickle', 'rb') as f:
            tmp = pickle.load(f)
            self.merges = tmp['merges']
            self.vocab = tmp['vocab']
    

def get_stats(ids, stats):
    for pair in zip(ids, ids[1:]):
        stats[pair] = stats.get(pair, 0) + 1

def merge(ids, pair, new_idx):
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i+1]) == pair:
            new_ids.append(new_idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids

if __name__ == '__main__':
    text_file = 'tiny_shakespeare.txt'
    vocab_size = 5000
    save_file = f'shake{vocab_size}'
    tok = Tokenizer()

    with open(text_file, 'r') as f:
        text = f.read()
    tok.train(text, vocab_size)

    tok.save(save_file)

    
    ###test

    #tok.load(save_file)
    #print(len(tok.vocab.items()))
    #print(text[:100])
    #print(tok.decode(tok.encode(text[:100])))
    #print(len(tok.encode(text[:500])))
    #print(len(text[:500].encode('utf-8')))
    #print(text[:500] == tok.decode(tok.encode(text[:500])))

