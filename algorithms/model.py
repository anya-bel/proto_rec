from io import open
import random
import time
import math

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from tqdm.auto import tqdm

from utils import EncoderRNN, EncoderCNN, WordEncoderCNN, AttnDecoderRNN, DecoderRNN, PrepDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token = 0
EOS_token = 1
PAD_token = 2


class ProtoReconstruction():
    """
    embeddings : lang for language embs in encoder, lang2 for language embs both in encoder and decoder, None for no language embs
    encoder : rnn, cnn, wordcnn
    decoder : rnn, attn
    """
    def __init__(self, embeddings, encoder, decoder):
        self.embeddings = embeddings
        self.encoder = encoder
        self.decoder = decoder
        self.proto_letter2index = {}
        self.proto_index2letter = {0: "SOS", 1: "EOS", 2: "PAD"}
        self.proto_n_letters = 3
        self.letter2index = {}
        self.index2letter = {0: "SOS", 1: "EOS", 2: "PAD"}
        self.n_letters = 3
        self.languages = []
        self.pairs = []
        self.valid = []
        self.test_pairs = []
        self.num_langs = 0
        self.hidden_size = 128
        self.max_length = 30
    
    @staticmethod
    def readLangs(file):
        """
        file: a path to a file with a dataset

        return: 75% of words are returned as the first list, 25% - as the second one (train and test sets)
        """

        print("Reading lines...")
        lines = open(file, encoding='utf-8').readlines()
        lines = [line.split('\t') for line in lines]
        langs = lines[0]
        pairs = []
        for i in range(len(lines[0])-1):
            cur_pairs = [[x[i+1].strip(), x[0].strip(), langs[i+1]] for x in lines[1:] if x[i+1].strip() != '-']
            pairs.extend(cur_pairs)
        length = int(len(pairs)/4)*3
        random.shuffle(pairs)
        return langs[1:], pairs[:length], pairs[length:]

    def addWord(self, word, lang):
        for letter in word:
                self.addLetter(letter, lang)

    def addLetter(self, letter, lang):
        if lang == 'proto':
            if letter not in self.proto_letter2index:
                self.proto_letter2index[letter] = self.proto_n_letters
                self.proto_index2letter[self.proto_n_letters] = letter
                self.proto_n_letters += 1
        else:
            if letter not in self.letter2index:
                self.letter2index[letter] = self.n_letters
                self.index2letter[self.n_letters] = letter
                self.n_letters += 1

    def prepareData(self, file):
        langs, pairs, test_pairs = self.readLangs(file)
        print("Read %s word pairs" % len(pairs))
        print("Counting letters...")
        for pair in pairs:
            self.addWord(pair[0], '')
            self.addWord(pair[1], 'proto')
        print("Counted letters:")
        print('input:', self.n_letters)
        print('proto:', self.proto_n_letters)
        return langs, pairs, test_pairs

    def tensorsFromPair(self, pair):
        ids = [self.letter2index[letter] for letter in pair[0]] + [EOS_token]
        #ids.extend([PAD_token for _ in range(self.max_length-len(ids))])
        tnsr = torch.tensor(ids, dtype=torch.long, device=device)

        proto_ids = [self.proto_letter2index[letter] for letter in pair[1]] + [EOS_token]
        #proto_ids.extend([PAD_token for _ in range(self.max_length-len(proto_ids))])
        proto_tnsr = torch.tensor(proto_ids, dtype=torch.long, device=device)

        lng_tnsr = torch.tensor([self.languages.index(pair[2])], dtype=torch.long, device=device)

        return (tnsr, proto_tnsr, lng_tnsr)

    @staticmethod
    def asMinutes(s):
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    def timeSince(self, since, percent):
        now = time.time()
        s = now - since
        es = s / (percent)
        rs = es - s
        return '%s (- %s)' % (self.asMinutes(s), self.asMinutes(rs))

    def getRoot(self, word, lang, hidden_size, encoder, decoder):
        with torch.no_grad():
            word, _, lang = self.tensorsFromPair([word, '', lang])

            enc_hidden = torch.zeros(1, 1, decoder.hidden_size, device=device)
            enc_outputs = torch.zeros(self.max_length, decoder.hidden_size, device=device)
            if isinstance(encoder, WordEncoderCNN):
                enc_outputs, _ = encoder(word.view(-1), lang.view(1, -1), enc_hidden)
            else:
                for i in range(word.size(0)):
                    enc_output, enc_hidden = encoder(word[i].view(1, -1), lang.view(1, -1), enc_hidden)
                    enc_outputs[i] = enc_output[0, 0]

            dec_input = torch.tensor([[SOS_token]], device=device)
            dec_hidden = enc_hidden
            output = list()
            for i in range(self.max_length):
                dec_output, dec_hidden = decoder(dec_input, lang.view(1, -1), dec_hidden, enc_outputs)
                values, ids = torch.max(dec_output, 1)
                output.append(self.proto_index2letter[int(ids[0])])
                dec_input = ids.view(-1, 1)
                if int(ids[0]) == EOS_token:
                    break
            output = "".join(output[:-1])
            return output

    def train_wo_batch(self, data, valid, encoder, decoder, hidden_size, iters=10, learning_rate=0.01):
        print("Training...")
        start = time.time()
        encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss(ignore_index=2)
        train_data = PrepDataset(data, self.tensorsFromPair)
        valid_data = PrepDataset(valid, self.tensorsFromPair)
        for iter in range(iters):
            indices = list(range(len(train_data)))
            random.shuffle(indices)
            progress_bar = tqdm(indices)
            for idx in progress_bar:
                word, root, lang = train_data[idx]
                loss = 0
                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()
                enc_hidden = torch.zeros(1, 1, decoder.hidden_size, device=device)
                enc_outputs = torch.zeros(self.max_length, decoder.hidden_size, device=device)
                if isinstance(encoder, WordEncoderCNN):
                    enc_outputs, _ = encoder(word.view(-1), lang.view(-1), None)
                else:
                    for i in range(word.size(0)):
                        enc_output, enc_hidden = encoder(word[i].view(-1), lang.view(-1), enc_hidden)
                        enc_outputs[i] = enc_output[0, 0]
                dec_input = torch.tensor([[SOS_token]], device=device)
                dec_hidden = enc_hidden
                for i in range(root.size(0)):
                    dec_output, dec_hidden = decoder(dec_input, lang.view(1, -1), dec_hidden, enc_outputs)
                    values, ids = torch.max(dec_output, 1)
                    dec_input = ids.view(-1, 1)
                    loss += criterion(dec_output, root[i].view(-1))
                    if dec_input.item() == EOS_token:
                        break
                loss.backward()
                encoder_optimizer.step()
                decoder_optimizer.step()
            valid_loss = []
            i = 0
            total = 0
            for word, root, lang in valid:
                try:
                    pred = self.getRoot(word, lang, hidden_size, encoder, decoder)
                except:
                    continue
                if pred == root:
                  i +=1
                total += 1
            #print(type(loss.item()))
            #print(loss.item()/self.max_length)
            current_loss = loss.item()/self.max_length
            accuracy = i/total
            print("loss: ", current_loss)
            print("accuracy: ", accuracy)
            progress_bar.set_postfix(current_loss=current_loss, accuracy=accuracy)

    def fit(self, file):
        self.languages, self.pairs, self.test_pairs = self.prepareData(file)
        length = int(len(self.pairs)*0.1)
        self.valid, self.pairs, = self.pairs[:length], self.pairs[length:]
        print('train size:', len(self.pairs))
        print('test size:', len(self.test_pairs))
        print('validation size: ', len(self.valid))

    def train(self, hidden_size=128, iters=10, learning_rate=0.01):
        self.hidden_size = hidden_size
        if self.encoder == 'rnn':
            self.encoder = EncoderRNN(self.n_letters, len(self.languages), self.hidden_size, self.embeddings).to(device)
        elif self.encoder == 'wordcnn':
            self.encoder = WordEncoderCNN(self.n_letters, len(self.languages), self.hidden_size, self.max_length).to(device)
        elif self.encoder == 'cnn':
            self.encoder = EncoderCNN(self.n_letters, len(self.languages), self.hidden_size).to(device)
        if self.decoder == 'rnn':
            if isinstance(self.encoder, EncoderCNN):
                hidden_size = int((self.hidden_size-2)/2)
                self.decoder = DecoderRNN(self.proto_n_letters, hidden_size, self.n_letters).to(device)
            else:
                self.decoder = DecoderRNN(self.proto_n_letters, self.hidden_size, self.n_letters).to(device)
        elif self.decoder == 'attn':
            if isinstance(self.encoder, EncoderCNN):
                hidden_size = int((self.hidden_size-2)/2)
                self.decoder = AttnDecoderRNN(self.proto_n_letters, hidden_size, len(self.languages), self.max_length, self.embeddings)
            else:
                self.decoder = AttnDecoderRNN(self.proto_n_letters, self.hidden_size, len(self.languages), self.max_length, self.embeddings)
        self.train_wo_batch(self.pairs, self.valid, self.encoder, self.decoder, self.hidden_size, iters=iters, learning_rate=learning_rate)