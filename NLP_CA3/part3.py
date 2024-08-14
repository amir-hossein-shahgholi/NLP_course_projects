import json
from collections import Counter
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


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


file_path = './data/train.json'
srl_data_handler = SRLDataHandler(file_path)
srl_data_handler.display_second_sentence_info()

class Vocab:
    def __init__(self, word2id=None):
        if word2id is None:
            self.word2id = {
                "<PAD>": 0,
                "<START>": 1,
                "<END>": 2,
                "<UNK>": 3
            }
        else:
            self.word2id = word2id
        self.id2word = {id: word for word, id in self.word2id.items()}

    def __getitem__(self, word):
        return self.word2id.get(word, self.word2id["<UNK>"])

    def __len__(self):
        return len(self.word2id)
    
    def add(self, word):
        if word not in self.word2id:
            index = len(self.word2id)
            self.word2id[word] = index
            self.id2word[index] = word
        return self.word2id[word]

    def word2indices(self, sents):
        return [[self[word] for word in sent] for sent in sents]

    def indices2words(self, word_ids):
        return [self.id2word[id_] for id_ in word_ids]

    @classmethod
    def to_input_tensor(cls, batch):
      sentences, verb_indices, labels = zip(*batch)
      sentences_padded = pad_sequence(sentences, batch_first=True, padding_value=vocab['<PAD>'])
      labels_padded = pad_sequence(labels, batch_first=True, padding_value=srl_data_handler.label_dict['O'])
      lengths = torch.tensor([len(sentence) for sentence in sentences])

      return sentences_padded, torch.tensor(verb_indices), labels_padded, lengths

    @classmethod
    def from_corpus(cls, corpus, size, freq_cutoff, remove_frac):
        word_freq = Counter(word for sent in corpus for word in sent)
        filtered_words = [word for word, freq in word_freq.items() if freq >= freq_cutoff]
        
        filtered_words = sorted(filtered_words, key=lambda x: -word_freq[x])
        if remove_frac > 0:
            num_remove = int(len(filtered_words) * remove_frac)
            filtered_words = filtered_words[:-num_remove]

        vocab = cls()
        for word in filtered_words[:size]:
            vocab.add(word)
        return vocab


class SRLModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(SRLModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(2 * hidden_dim, output_dim)

    def forward(self, x, verb_indices, lengths):
        embedded = self.embedding(x)
        packed_embedded = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        gru_out, _ = self.gru(packed_embedded)
        gru_out, _ = pad_packed_sequence(gru_out, batch_first=True)
        verb_hidden = gru_out[range(len(gru_out)), verb_indices]
        verb_hidden_expanded = verb_hidden.unsqueeze(1).expand(-1, gru_out.shape[1], -1)
        combined_output = torch.cat((gru_out, verb_hidden_expanded), dim=2)
        dense_outputs = self.fc(combined_output)
        return dense_outputs


def train_model(model, train_loader, valid_loader, num_epochs, learning_rate):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min')
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        train_loss_accum = 0
        for words, verbs, labels, lengths in train_loader:
            optimizer.zero_grad()
            outputs = model(words, verbs, lengths)
            loss = criterion(outputs.transpose(1, 2), labels)
            loss.backward()
            optimizer.step()
            train_loss_accum += loss.item() * words.size(0)
        
        avg_train_loss = train_loss_accum / len(train_loader.dataset)
        train_losses.append(avg_train_loss)
        
        model.eval()
        val_loss_accum = 0
        with torch.no_grad():
            for words, verbs, labels, lengths in valid_loader:
                outputs = model(words, verbs, lengths)
                loss = criterion(outputs.transpose(1, 2), labels)
                val_loss_accum += loss.item() * words.size(0)
                
        avg_val_loss = val_loss_accum / len(valid_loader.dataset)
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
    return model, train_losses, val_losses

class SRLLoader(Dataset):
    def __init__(self, sentences, verb_indices, labels, vocab, label_dict):
        self.sentences = [torch.tensor([vocab[word] for word in sentence]) for sentence in sentences]
        self.verb_indices = torch.tensor(verb_indices)
        self.labels = [torch.tensor([label_dict[label] for label in ls]) for ls in labels]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx], self.verb_indices[idx], self.labels[idx]

def prepare_dataloader(text, verbs, labels, vocab, label_dict, batch_size=64):
    dataset = SRLLoader(text, verbs, labels, vocab, label_dict)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=Vocab.to_input_tensor)
    return loader

test_file_path = './data/test.json'
valid_file_path = './data/valid.json'

test_data_handler = SRLDataHandler(test_file_path)
valid_data_handler = SRLDataHandler(valid_file_path)

train_text = srl_data_handler.data['text']
train_verbs = srl_data_handler.data['verb_index']
train_labels = srl_data_handler.data['srl_frames']

valid_text = valid_data_handler.data['text']
valid_verbs = valid_data_handler.data['verb_index']
valid_labels = valid_data_handler.data['srl_frames']

vocab = Vocab.from_corpus(train_text, size=20000, freq_cutoff=2, remove_frac=0.3)
label_dict = srl_data_handler.label_dict

train_loader = prepare_dataloader(train_text, train_verbs, train_labels, vocab, label_dict, batch_size=64)
valid_loader = prepare_dataloader(valid_text, valid_verbs, valid_labels, vocab, label_dict, batch_size=64)

vocab_size = len(vocab)
emb_dim = 64
hidden_dim = 64
output_dim = len(srl_data_handler.label_dict)
num_epochs = 10
learning_rate = 0.01

model = SRLModel(vocab_size, emb_dim, hidden_dim, output_dim)
model, train_losses, val_losses = train_model(model, train_loader, valid_loader, num_epochs, learning_rate)

plt.figure()
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

def detailed_f1_scores(model, data_loader, label_dict):
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for sentences, verb_indices, labels, lengths in data_loader:
            outputs = model(sentences, verb_indices, lengths)
            predictions = torch.argmax(outputs, dim=2)
            for i, length in enumerate(lengths):
                all_preds.extend(predictions[i, :length].tolist())
                all_labels.extend(labels[i, :length].tolist())
    
    label_names = {idx: label for label, idx in label_dict.items()}
    report = classification_report(all_labels, all_preds, labels=list(label_dict.values()), target_names=list(label_names.values()), zero_division=0)
    return report
train_report = detailed_f1_scores(model, train_loader, srl_data_handler.label_dict)
print("Train Classification Report:")
print(train_report)

def calculate_accuracy(model, data_loader):
    correct, total = 0, 0
    model.eval()
    with torch.no_grad():
        for sentences, verb_indices, labels, lengths in data_loader: 
            outputs = model(sentences, verb_indices, lengths)
            predictions = torch.argmax(outputs, dim=2)
            for i, length in enumerate(lengths):
                total += length
                correct += (predictions[i, :length] == labels[i, :length]).sum().item()
    accuracy = correct / total if total > 0 else 0
    return accuracy

train_accuracy = calculate_accuracy(model, train_loader)
valid_accuracy = calculate_accuracy(model, valid_loader)
print(f'Train Accuracy: {train_accuracy * 100:.4f}%')
print(f'Validation Accuracy: {valid_accuracy * 100:.4f}%')