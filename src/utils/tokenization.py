from collections import defaultdict

class BPE:
    def __init__(self, vocab_size, tokens, indexes) -> None:
        self.vocab_size = vocab_size
        
        self.PAD_TOKEN = tokens[0]
        self.SOS_TOKEN = tokens[1]
        self.EOS_TOKEN = tokens[2]
        self.UNK_TOKEN = tokens[3]

        self.PAD_INDEX = indexes[0]
        self.SOS_INDEX = indexes[1]
        self.EOS_INDEX = indexes[2]
        self.UNK_INDEX = indexes[3]

        self.toks_and_inds = {self.PAD_TOKEN: self.PAD_INDEX, 
                              self.SOS_TOKEN: self.SOS_INDEX, 
                              self.EOS_TOKEN: self.EOS_INDEX, 
                              self.UNK_TOKEN: self.UNK_INDEX}
        
        self.vocabs = None
        self.merges = {}
        
        
    def word_freqs(self, sentences):
        word_freqs = defaultdict(int)
        for sentence in sentences:
            segments = sentence.split()
            for word in segments:
                word_freqs[word] += 1
        return word_freqs
    
    def alphabet(self, word_freqs):
        alphabet = []
        for word in word_freqs.keys():
            for letter in word:
                alphabet.append(letter)
        alphabet.sort()
        return alphabet
    
    def compute_pair_freqs(self, word_freqs, splits):
        pair_freqs = defaultdict(int)
        for word, freq in word_freqs.items():
            split = splits[word]
            if len(split) == 1:
                continue
            for i in range(len(split) - 1):
                pair = (split[i], split[i+1])
                pair_freqs[pair] += freq
        return pair_freqs
    
    def merge_pair(self, a, b, word_freqs, splits):
        for word in word_freqs:
            split = splits[word]
            if len(split) == 1:
                continue
            
            i = 0
            while i < len(split) - 1:
                if split[i] == a and split[i+1] == b:
                    split = split[:i] + [a+b] + split[i+2:]
                else:
                    i += 1
            splits[word] = split
        return splits
    
    def build_vocab(self, sentences):
        self.vocabs = self.toks_and_inds.copy()
        
        word_freqs = self.word_freqs(sentences)
        alphabeta = self.alphabet(word_freqs)
        for letter in alphabeta:
            self.vocabs[letter] = len(self.vocabs)
        
        splits = {word: [c for c in word] for word in word_freqs.keys()}
        
        while len(self.vocabs) < self.vocab_size:
            pair_freqs = self.compute_pair_freqs(word_freqs, splits)
            
            best_pair = ""
            max_freq = None
            for pair, freq in pair_freqs.items():
                if max_freq is None or max_freq < freq:
                    best_pair = pair
                    max_freq = freq
            if freq == 1:
                break
            splits = self.merge_pair(*best_pair, word_freqs, splits)
            self.merges[best_pair] = best_pair[0] + best_pair[1]
            self.vocabs[best_pair[0]+best_pair[1]] = len(self.vocabs)
            
    def tokenize(self, text):
        words = text.split()
        
        splits = [[letter for letter in word] for word in words]
        
        for pair, merge in self.merges.items():
            for idx, split in enumerate(splits):
                i = 0
                while i < len(split) - 1:
                    if split[i] == pair[0] and split[i+1] == pair[1]:
                        split = split[:i] + [merge] + split[i+2:]
                    else:
                        i += 1
                splits[idx] = split
        return sum(splits, [])
    
class BBPE:
    def __init__(self, vocab_size, tokens, indexes) -> None:
        self.vocab_size = vocab_size
        self.vocabs = None
        self.merges = {}
        
        self.PAD_TOKEN = tokens[0]
        self.SOS_TOKEN = tokens[1]
        self.EOS_TOKEN = tokens[2]
        self.UNK_TOKEN = tokens[3]

        self.PAD_INDEX = indexes[0]
        self.SOS_INDEX = indexes[1]
        self.EOS_INDEX = indexes[2]
        self.UNK_INDEX = indexes[3]

        self.toks_and_inds = {self.PAD_TOKEN: self.PAD_INDEX, 
                              self.SOS_TOKEN: self.SOS_INDEX, 
                              self.EOS_TOKEN: self.EOS_INDEX, 
                              self.UNK_TOKEN: self.UNK_INDEX}
        
    def build_vocab(self, sentences):
        self.vocabs = self.toks_and_inds.copy()
        
        word_freqs = self.word_freqs(sentences)
        initial_vocab = [bytes([byte]) for byte in range(256)]
        
        for bt in initial_vocab:
            self.vocabs[bt] = len(self.vocabs)
            
        splits = {word: [bytes([byte]) for byte in word] for word in word_freqs.keys()}
        
        while len(self.vocabs) < self.vocab_size:
            pair_freqs = self.compute_pair_freqs(word_freqs, splits)
            best_pair = ""
            max_freq = None
            for pair, freq in pair_freqs.items():
                if max_freq is None or max_freq < freq:
                    best_pair = pair
                    max_freq = freq
            if freq == 1:
                break
            
            splits = self.merge_pair(best_pair, word_freqs, splits)
            if best_pair != "":
                self.merges[best_pair] = best_pair[0] + best_pair[1]
                self.vocabs[best_pair[0] + best_pair[1]] = len(self.vocabs)
    def word_freqs(self, sentences):
        word_freqs = defaultdict(int)
        
        for sentence in sentences:
            segments = sentence.split()
            for word in segments:
                word_freqs[word.encode('utf-8')] += 1
        return word_freqs
    
    def compute_pair_freqs(self, word_freqs, splits):
        pair_freqs = defaultdict(int)
        
        for word, freq in word_freqs.items():
            split = splits[word]
            if len(split) == 1:
                continue
            for i in range(len(split) - 1):
                pair = (split[i], split[i+1])
                pair_freqs[pair] += freq
        return pair_freqs
    
    def merge_pair(self, pair, word_freqs, splits):
        merged_byte = pair[0] + pair[1] if pair != "" else ""
        
        for word in word_freqs:
            split = splits[word]
            if len(split) == 1:
                continue
            
            i = 0
            while i < len(split) - 1:
                if split[i] == pair[0] and split[i+1] == pair[1]:
                    split = split[:i] + [merged_byte] + split[i+2:]
                else:
                    i += 1
            splits[word] = split
        return splits
        
if __name__=="__main__":
    corpus = [
    "This is the Hugging Face Course.",
    "This chapter is about tokenization.",
    "This section shows several tokenizer algorithms.",
    "Hopefully, you will be able to understand how they are trained and generate tokens.",
    ]
    
    PAD_TOKEN = '<pad>'
    SOS_TOKEN = '<sos>'
    EOS_TOKEN = '<eos>'
    UNK_TOKEN = '<unk>'

    PAD_INDEX = 0
    SOS_INDEX = 1
    EOS_INDEX = 2
    UNK_INDEX = 3

    tokens  = (PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN)
    indexes = (PAD_INDEX, SOS_INDEX, EOS_INDEX, UNK_INDEX)

    bpe = BBPE(vocab_size=500, 
                tokens=tokens, indexes=indexes)
    
    bpe.build_vocab(corpus)
    print(bpe.vocabs)
    