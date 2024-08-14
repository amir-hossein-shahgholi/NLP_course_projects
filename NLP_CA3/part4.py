import json
import torch
import torch.nn as nn
import random
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import Adam
from collections import defaultdict
import itertools
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class SRLDataHandler:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = self.load_data(file_path)
        self.label_dict = {
            'O': 0, 'B-ARG0': 1, 'I-ARG0': 2, 'B-ARG1': 3, 'I-ARG1': 4, 'B-ARG2': 5,
            'I-ARG2': 6, 'B-ARGM-LOC': 7, 'I-ARGM': 8, 'B-ARGM-TMP': 9, 'I-ARGM-TMP': 10
        }

    def load_data(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data

    def display_second_sentence_info(self):
        print("________________________")
        print(f"Text: {self.data['text'][1]}")
        print(f"srl_frames: {self.data['srl_frames'][1]}")
        print(f"verb_index: {self.data['verb_index'][1]}")
        print(f"words_indices: {self.data['words_indices'][1]}")
        print(f"Labeled srl_frames: {self.convert_labels_to_numeric(self.data['srl_frames'][1])}")
        print("________________________")

    def convert_labels_to_numeric(self, labels):
        return [self.label_dict[label] for label in labels]
    
    def pad_sentences(self, max_length):
        padded_text = []
        for sentence in self.data['text']:
            padded_sentence = sentence + [self.pad_token] * (max_length - len(sentence))
            padded_text.append(padded_sentence[:max_length])
        self.data['text'] = padded_text

file_path = '/content/drive/MyDrive/nlp/data/train.json'
srl_data_handler = SRLDataHandler(file_path)
srl_data_handler.display_second_sentence_info()

def generate_qa_pairs(data, labels_desc):
    qa_pairs = []
    for idx, sentence in enumerate(data['text']):
        for verb_pos in [data['verb_index'][idx]]:
            verb = sentence[verb_pos]
            frame = data['srl_frames'][idx]
            for label in labels_desc:
                question = f"{verb} [SEPT] {' '.join(sentence)}. {label}"
                answer_indices = [i for i, label_tag in enumerate(frame) if label_tag.endswith(label)]
                if answer_indices:
                    span = ' '.join(sentence[min(answer_indices):max(answer_indices)+1])
                    answer = f"<s> {span} </s>"
                else:
                    answer = "<s> </s>"
                qa_pairs.append({'input': question, 'output': answer})
    return qa_pairs
new_label_dict = {'ARG0': 0, 'ARG1': 1, 'ARG2': 2, 'ARGM-LOC': 3, 'ARGM-TMP': 4}
qa_dataset = generate_qa_pairs(srl_data_handler.data, new_label_dict.keys())
for i in qa_dataset[:10]:
  print(i)

valid_file_path = '/content/drive/MyDrive/nlp/data/valid.json'
valid_data_handler = SRLDataHandler(valid_file_path)
valid_qa_dataset = generate_qa_pairs(valid_data_handler.data, new_label_dict.keys())

# !wget http://nlp.stanford.edu/data/glove.6B.zip
# !unzip glove.6B.zip -d glove.6B

embeddings_index = {}
with open('glove.6B/glove.6B.300d.txt', 'r', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float32")
        embeddings_index[word] = coefs

print("Found %s word vectors in GloVe embeddings." % len(embeddings_index))

def create_embedding_matrix(word_index, embeddings, embedding_dim=300):
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

embeddings = embeddings_index
def extract_vocabulary(qa_pairs):
    word_count = defaultdict(int)
    for pair in qa_pairs:
        words_in_input = pair['input'].split()
        for word in words_in_input:
            word_count[word] += 1
    sorted_words = sorted(word_count.keys(), key=lambda x: -word_count[x])
    word_index = {word: idx + 1 for idx, word in enumerate(sorted_words)}
    return word_index
train_word_index = extract_vocabulary(qa_dataset)
valid_word_index = extract_vocabulary(valid_qa_dataset)
all_words = set(itertools.chain(train_word_index.keys(), valid_word_index.keys()))
combined_word_index = {word: idx+1 for idx, word in enumerate(sorted(all_words))}
embedding_matrix = create_embedding_matrix(combined_word_index, embeddings)
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout, embedding_matrix=None):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, emb_dim)
        
        if embedding_matrix is not None:
            self.embedding.weight = nn.Parameter(torch.from_numpy(embedding_matrix).float(), requires_grad=False)
        
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, bidirectional=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)  
        return outputs, hidden, cell
class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        src_len = encoder_outputs.shape[0]
        if hidden.dim() == 3:
            hidden = hidden[-1]

        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)

        return torch.softmax(attention, dim=1)

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, n_layers, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(input_size=(enc_hid_dim * 2) + emb_dim, hidden_size=dec_hid_dim, num_layers=n_layers, dropout=dropout)
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.attention = attention
        self.n_layers = n_layers 
    def forward(self, input, hidden, cell, encoder_outputs):
        input = input.unsqueeze(0)
        
        embedded = self.dropout(self.embedding(input))
        hidden = hidden[-1].unsqueeze(0)
        cell = cell[-1].unsqueeze(0)
        a = self.attention(hidden.squeeze(0), encoder_outputs)
        a = a.unsqueeze(1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        weighted = torch.bmm(a, encoder_outputs)
        weighted = weighted.permute(1, 0, 2)
        print("embedded size:", embedded.size())
        print("weighted size:", weighted.size())
        
        rnn_input = torch.cat((embedded, weighted), dim=2)

        print(rnn_input.size())
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))
        
        return prediction, hidden, cell
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        encoder_outputs, hidden, cell = self.encoder(src)
        input = trg[0,:]

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1) 
            input = trg[t] if teacher_force else top1
        
        return outputs
class QADataset(Dataset):
    def __init__(self, qa_pairs, word_vocab, output_vocab):
        self.qa_pairs = qa_pairs
        self.word_vocab = word_vocab
        self.output_vocab = output_vocab

    def __len__(self):
        return len(self.qa_pairs)

    def __getitem__(self, idx):
        qa_pair = self.qa_pairs[idx]
        src = torch.tensor([self.word_vocab.get(word, 0) for word in qa_pair['input'].split()], dtype=torch.long)
        trg = torch.tensor([self.output_vocab.get(word, 0) for word in qa_pair['output'].split()], dtype=torch.long)
        return src, trg

    @staticmethod
    def collate_fn(batch):
        srcs, trgs = zip(*batch)
        src_padded = pad_sequence(srcs, padding_value=0)
        trg_padded = pad_sequence(trgs, padding_value=0)
        return src_padded, trg_padded

def extract_vocabulary(qa_pairs):
    word_count = defaultdict(int)
    for pair in qa_pairs:
        for word in pair['input'].split() + pair['output'].split():
            word_count[word] += 1
    return {
        word: i + 1  
        for i, word in enumerate(word_count)
    }

train_vocab = extract_vocabulary(qa_dataset)
output_vocab = extract_vocabulary(qa_dataset + valid_qa_dataset)

input_dim = len(train_vocab) + 1
output_dim = len(output_vocab) + 1

emb_dim = 300  
hid_dim = 512  
n_layers = 2   
dropout = 0.5  

encoder = Encoder(input_dim, emb_dim, hid_dim, n_layers, dropout)
attention_layer = Attention(hid_dim, hid_dim)
decoder = Decoder(output_dim, emb_dim, hid_dim * 2, hid_dim, n_layers, dropout, attention_layer)
model = Seq2Seq(encoder, decoder, device).to(device)

optimizer = Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss(ignore_index=0) 

def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0

    for i, (src, trg) in enumerate(iterator):
        src = src.to(device)
        trg = trg.to(device)
        optimizer.zero_grad()
        
        output = model(src, trg)
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)
train_dataset = QADataset(qa_dataset, train_vocab, output_vocab)
train_dataloader = DataLoader(train_dataset, batch_size=32, collate_fn=QADataset.collate_fn, shuffle=True)
train_losses = []

epochs = 10
clip = 1
for epoch in range(epochs):
    train_loss = train(model, train_dataloader, optimizer, criterion, clip)
    train_losses.append(train_loss)  
    print(f'Epoch {epoch+1} Training Loss: {train_loss:.3f}')
    torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pth')

plt.figure()
plt.plot(range(1, epochs+1), train_losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.legend()
plt.show()

