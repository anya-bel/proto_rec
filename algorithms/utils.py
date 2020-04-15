import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

class EncoderRNN(nn.Module):
    def __init__(self, n_letters, lang_num, hidden_size, embeddings=None):
        super(EncoderRNN, self).__init__()
        self.word_embedding = nn.Embedding(n_letters, hidden_size)
        self.lang_embedding = nn.Embedding(lang_num, hidden_size)
        self.lang_gru = nn.GRU(hidden_size*2, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.embeddings = embeddings

    def forward(self, word, lang, hidden):
        word_embedded = self.word_embedding(word).view(1, 1, -1)
        if self.embeddings:
            lang_embedded = self.lang_embedding(lang).view(1, 1, -1)
            output, hidden = self.lang_gru(torch.cat((word_embedded, lang_embedded), -1), hidden)
        else:
            output, hidden = self.lang_gru(word_embedded, hidden)
        return output, hidden

class EncoderCNN(nn.Module):
    def __init__(self, n_letters, lang_num, hidden_size):
        super(EncoderCNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(n_letters, hidden_size)
        self.conv = nn.Conv1d(1, 1, 3)
        self.pool = nn.MaxPool1d(2, 2)

    def forward(self, input, lang, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output = self.pool(self.conv(output))
        return output, output

class WordEncoderCNN(nn.Module):
    def __init__(self, n_letters, lang_num, hidden_size, max_length):
        super(WordEncoderCNN, self).__init__()
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.embedding = nn.Embedding(n_letters, 128)
        self.conv1 = nn.Conv1d(hidden_size, hidden_size, 3, padding=1)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, 3, padding=1)

    def forward(self, input, lang, hidden):
        extention = torch.tensor([2 for i in range(self.max_length-input.size(0))], device=device)
        input = torch.cat((input, extention), 0)
        input = input.view(1, 1, -1)
        embedded = self.embedding(input).view(1, -1, self.hidden_size).transpose(2,1)
        output = self.conv1(embedded)
        output = self.conv2(output)
        output = output.transpose(2,1)
        output = output[0]
        return output, output

class DecoderRNN(nn.Module):
    def __init__(self,  output_size, hidden_size, n_letters):
        super(DecoderRNN, self).__init__()
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size
        self.embedding =  nn.Embedding(n_letters, hidden_size)

    def forward(self, input, lang, hidden, ennoder_outputs):
        input = self.embedding(input)
        output, hidden = self.gru(input, hidden)
        output = self.out(output[:,-1])
        return output, hidden

class AttnDecoderRNN(nn.Module):
    def __init__(self, output_size, hidden_size, lang_num, max_length, embeddings=None, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.lang_embedding = nn.Embedding(lang_num, self.hidden_size)
        self.lang_gru = nn.GRU(self.hidden_size*2, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.embeddings = embeddings

    def forward(self, input, lang, hidden, encoder_outputs):
        #embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.embedding(input)
        if self.embeddings == 'lang2':
          lang_embedded = self.lang_embedding(lang)
          embedded, lang_hidden = self.lang_gru(torch.cat((embedded, lang_embedded), -1))
        embedded = self.dropout(embedded)
        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden

class PrepDataset(torch.utils.data.Dataset):
    def __init__(self, data, tensorsFromPair): 
        super(PrepDataset).__init__()
        self.data = [tensorsFromPair(x) for x in data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

def minimumEditDistance(s1,s2):
    if len(s1) > len(s2):
        s1,s2 = s2,s1
    distances = range(len(s1) + 1)
    for index2,char2 in enumerate(s2):
        newDistances = [index2+1]
        for index1,char1 in enumerate(s1):
            if char1 == char2:
                newDistances.append(distances[index1])
            else:
                newDistances.append(1 + min((distances[index1],
                                             distances[index1+1],
                                             newDistances[-1])))
        distances = newDistances
    return distances[-1]

