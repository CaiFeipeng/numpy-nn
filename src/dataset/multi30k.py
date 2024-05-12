import numpy as np
import os
from collections import defaultdict

class Multi30k:
    def __init__(self, data_root, tokens, indexes, batch_size=1, min_freq=10):
        self.data_root=data_root
        self.batch_size = batch_size
        self.min_freq = min_freq
        
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
        
        self.train, self.val, self.test = self.load_samples()
        self.train_idx, self.val_idx, self.test_idx = self.preprocess()

    def load_samples(self):
        ret = []
        datasets = ["train", "val", "test"]

        for dataset in datasets:

            examples = []

            en_path = os.path.join(self.data_root, dataset + '.en')
            de_path = os.path.join(self.data_root, dataset + '.de')

            en_file = [l.strip() for l in open(en_path, 'r', encoding='utf-8')]
            de_file = [l.strip() for l in open(de_path, 'r', encoding='utf-8')]

            assert len(en_file) == len(de_file)

            for i in range(len(en_file)):
                if en_file[i] != '' and de_file[i] != '':
                    en_seq, de_seq = en_file[i], de_file[i]

                    examples.append({'en': en_seq, 'de': de_seq})
        
            ret.append(examples)
   
        return tuple(ret)
    def preprocess(self):

        self.clear_dataset()
        print(f"train data sequences num = {len(self.train)}")

        self.vocabs = self.build_vocab()
        print(f"EN vocab length = {len(self.vocabs[0])}; DE vocab length = {len(self.vocabs[1])}")

        train_data = self.add_tokens(self.train, self.batch_size)
        print(f"batch num = {len(train_data)}")

        train_source, train_target = self.build_dataset(train_data, self.vocabs)

        test_data = self.add_tokens(self.test, self.batch_size)
        test_source, test_target = self.build_dataset(test_data, self.vocabs)

        val_data = self.add_tokens(self.val, self.batch_size)
        val_source, val_target = self.build_dataset(val_data, self.vocabs)

        return (train_source, train_target), (val_source, val_target), (test_source, test_target)
    
    def get_vocabs(self):
        return self.vocabs

    def filter_seq(self, seq):
        chars2remove = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'

        return ''.join([c for c in seq if c not in chars2remove])

    def lowercase_seq(self, seq):
        return seq.lower()
    
    def clear_dataset(self):
        for dataset in self.train, self.val, self.test:
            for example in dataset:
                example['en'] = self.filter_seq(example['en'])
                example['de'] = self.filter_seq(example['de'])

                example['en'] = self.lowercase_seq(example['en'])
                example['de'] = self.lowercase_seq(example['de'])

                example['en'] = example['en'].split()
                example['de'] = example['de'].split()

    def build_vocab(self):
        en_vocab = self.toks_and_inds.copy(); en_vocab_freqs = defaultdict(int)
        de_vocab = self.toks_and_inds.copy(); de_vocab_freqs = defaultdict(int)

        for example in self.train:
            for word in example['en']:
                en_vocab_freqs[word] += 1
            for word in example['de']:
                de_vocab_freqs[word] += 1

        for example in self.train:
            for word in example['en']:
                if word not in en_vocab and en_vocab_freqs[word] >= self.min_freq:
                    en_vocab[word] = len(en_vocab)
            for word in example['de']:
                if word not in de_vocab and de_vocab_freqs[word] >= self.min_freq:
                    de_vocab[word] = len(de_vocab)

        return en_vocab, de_vocab   
    
    def add_tokens(self, dataset, batch_size):
        for example in dataset:
            example['en'] = [self.SOS_TOKEN] + example['en'] + [self.EOS_TOKEN]
            example['de'] = [self.SOS_TOKEN] + example['de'] + [self.EOS_TOKEN]
            
        data_batches = np.array_split(dataset, np.arange(batch_size, len(dataset), batch_size))

        for batch in data_batches:
            max_en_seq_len, max_de_seq_len = 0, 0

            for example in batch:
                max_en_seq_len = max(max_en_seq_len, len(example['en']))
                max_de_seq_len = max(max_de_seq_len, len(example['de']))

            for example in batch:
                example['en'] = example['en'] + [self.PAD_TOKEN] * (max_en_seq_len - len(example['en']))
                example['de'] = example['de'] + [self.PAD_TOKEN] * (max_de_seq_len - len(example['de']))


        return data_batches
    
    def build_dataset(self, dataset, vocabs):
        
        source, target = [], []
        for batch in dataset:
            
            source_tokens, target_tokens = [], []
            for example in batch:
                en_inds = [vocabs[0][word] if word in vocabs[0] else self.UNK_INDEX for word in example['en']]
                de_inds = [vocabs[1][word] if word in vocabs[1] else self.UNK_INDEX for word in example['de']]
                
                source_tokens.append(en_inds)
                target_tokens.append(de_inds)

            source.append(np.asarray(source_tokens))
            target.append(np.asarray(target_tokens))

        return source, target     
    def __len__(self):
        return len(self.train_idx[0])

    def __getitem__(self, index):
        source = self.train_idx[0][index]
        target = self.train_idx[1][index]
        return source, target        

if __name__=='__main__':
    DATA_TYPE = np.float32
    BATCH_SIZE = 32

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

    dataloader = Multi30k(data_root='./data/multi30k',
                          tokens=tokens, indexes=indexes, batch_size=10)
    for idx, (s,t) in enumerate(dataloader):
        print(idx)