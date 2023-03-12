import re
import sys 
import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
import math
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)

corpus = sys.argv[1]

def read_file():
    with open(corpus, 'r') as f:
        data = f.readlines()
    return data

def sent_tokenization(data):
    # realised that readlines isn't enough to tokenize the data into sequences of sentences because of uneven spacing present throughtout the text - tried my best to tokenize properly - still have doubts about the accuracy of the tokenization
    big_data = ""

    for sentence in data:
        sentence = re.sub(r'\n', ' ', sentence)
        if sentence != " ":
            big_data += sentence
    
    sentences = []

    big_data = big_data.strip()
    big_data = " " + big_data + " "
    
    # taking care of sentences where they aren't really ending because of words like Mr or Mrs or Ms or any other word that starts with a capital letter
    big_data = re.sub(r'(Mr|Ms|Mrs)\.', r'\1<prd>', big_data)

    # converting all things in the form of an url to the URL tag - so that websites are addressed in a uniform manner
    big_data = re.sub(r'\w+:\/\/\S+', '<URL>', big_data)

    # taking care of numbers with decimals
    big_data = re.sub(r'([0-9])\.([0-9])', r'\1<prd>\2', big_data)

    # taking care of ellipses in the text if any
    big_data = re.sub(r'\.\+', ' ', big_data)

    # taking care of names such as R. B. William and all
    big_data = re.sub(r'\s([A-Z])\.\ ', r' \1<prd>\ ', big_data)

    # convering the words to denote italics to remove the italics underscores
    big_data = re.sub(r'\_(\w+)\_', r'\1', big_data)

    big_data = re.sub(r'(\w)\.(\w)\.(\w)\.', r'\1<prd>\2<prd>\3<prd>', big_data)
    big_data = re.sub(r'(\w)\.(\w)\.', r'\1<prd>\2<prd>', big_data)
    big_data = re.sub(r'\ (\w)\.', r' \1<prd>', big_data)

    if "\"" in big_data: 
        big_data = big_data.replace(".\"","\".")
    
    if "!" in big_data:
        big_data = big_data.replace("!\"","\"!")
    
    if "?" in big_data:
        big_data = big_data.replace("?\"","\"?")
    
    big_data = big_data.replace(".",".<stop>")
    big_data = big_data.replace("?","?<stop>")
    big_data = big_data.replace("!","!<stop>")

    big_data = big_data.replace("<prd>",".")

    sentences = big_data.split("<stop>")
    sentences = sentences[:-1]

    sentences = [s.strip() for s in sentences]

    return sentences

def tokenization(text):
    # this is a function which will tokenize the sentences and actually give us a list of words in the sentence 
    clean_text = text

    # converting the text to lower case to introduce uniformity
    clean_text = clean_text.lower()

    # converting all things in the form of a hashtag to the HASHTAG tag
    clean_text = re.sub(r'\#\w+', '<HASHTAG>', clean_text)

    # converting all things in the form of a mention to the MENTION tag
    clean_text = re.sub(r'\@\w+', '<MENTION>', clean_text)

    # removing the chapter numbers from text - this is a very special case and is sort of a text cleaning work that is being done in this case for the given piece of text
    clean_text = re.sub(r'\[ [0-9]+ \]', '', clean_text)

    # convering anything that is a number to the number tag
    clean_text = re.sub(r'\d+\.*\d*', '<NUM>', clean_text)

    # convering things in the form of NUM(st, nd, rd, th) to just NUM - improving on the num tag
    clean_text = re.sub(r'<NUM>(?:st|nd|rd|th)', '<NUM>', clean_text)

    # taking care of &c that is seen in a number of places - don't really know what it does, so removing it here 
    clean_text = re.sub(r'(?:\.|\,)\ \&c', '', clean_text)

    # taking care of words with 'll form 
    clean_text = re.sub(r'\'ll\ ', ' will ', clean_text)

    # taking care of the word can't
    clean_text = re.sub(r'\ can\'t\ ', 'can not', clean_text)
    clean_text = re.sub(r'\ won\'t\ ', 'would not', clean_text)

    # taking care of words with n't form 
    clean_text = re.sub(r'n\ t\ ', ' not ', clean_text)

    # taking care of words with 're form 
    clean_text = re.sub(r'\'re\ ', ' are ', clean_text)

    # taking care of words with 'm form 
    clean_text = re.sub(r'\'m\ ', ' am ', clean_text)

    # taking care of the 've form 
    clean_text = re.sub(r'\'ve\ ', ' have ', clean_text)

    # taking care of the 'd form
    clean_text = re.sub(r'\'d\ ', ' would ', clean_text)

    # didn't really do anything for the 's form cause it might denote possession and in this corpus it is more common for 's to denote possession - so just separated the 's forms 
    clean_text = re.sub(r'\'s\ ', ' \'s', clean_text)

    # hyphenated words - while playing around with the corpus, realised it'd make more sense to combine them than to break them down due to presence of words like head-ache or to-day - doing it twice for words like mother-in-law : tried fancy stuff, didn't really work out ono
    clean_text = re.sub(r'(\w+)\-(\w+)', r'\1\2', clean_text)
    clean_text = re.sub(r'(\w+)\-(\w+)', r'\1\2', clean_text)

    # excess hypens need to go now 
    clean_text = re.sub(r'\-+', ' ', clean_text)

    # padding punctuation characters to ensure cute tokenization
    # clean_text = re.sub(r'\s*([,.?!;:"()])\s*', r' \1 ', clean_text)

    # so in some n gram models, punctuation is included while in some it is not (like google's model), and we can also choose to remove it completely - we would just need to change the substitution string
    clean_text = re.sub(r'\s*([,.?!;:"()—_\\])\s*', r' ', clean_text)

    # getting rid of all the extra spaces that there might be present
    clean_text = re.sub(r'\s+', ' ', clean_text)

    # getting rid of trainling spaces 
    clean_text.strip()

    tokens = [token for token in clean_text.split() if token != ""]
    # tokens = ['<s>'] + tokens + ['</s>']
    
    return tokens

vocab = set()
word_count = {}

def count_words(clean_sentences):
    for sentence in clean_sentences:
        for word in sentence:
            if word in word_count:
                word_count[word] += 1
            else:
                word_count[word] = 1

def create_vocabulary():
    for word in word_count:
        if word_count[word] >= 1:
            vocab.add(word)
    
    global vocabulary
    vocabulary = list(vocab)

    vocabulary.insert(0, '<unk>')
    vocabulary.insert(1, '<eos>')

    return vocabulary

def words_and_indices(vocabulary):
    # this function creates a dictionary of words and their corresponding indices
    global words_to_indices, indices_to_words
    words_to_indices = {}
    indices_to_words = {}

    for index, word in enumerate(vocabulary):
        words_to_indices[word] = index
        indices_to_words[index] = word

def get_data(dataset, vocabulary, batch_size):
    data = []
    for sentence in dataset:
        sentence.append('<eos>')
        sentence = [words_to_indices[word] if word in vocabulary else words_to_indices['<unk>'] for word in sentence]
        data.extend(sentence)
    
    data = torch.LongTensor(data)
    num_batches = data.shape[0] // batch_size
    data = data[:num_batches * batch_size]
    data = data.view(batch_size, -1)

    return data

class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout_rate, tie_weights):

        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers = num_layers, dropout = dropout_rate, batch_first = True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim, vocab_size)

        if tie_weights:
            assert embedding_dim == hidden_dim, "Embedding dimension and hidden dimension must be equal for tied weights"
            self.embedding.weight = self.fc.weight
        self.init_weights()

    def forward(self, src, hidden):
        embedding = self.dropout(self.embedding(src))
        output, hidden = self.lstm(embedding, hidden)
        output = self.dropout(output)
        prediction = self.fc(output)
        return prediction, hidden

    def init_weights(self):
        init_range_embed = 0.1
        init_range_other = 1/math.sqrt(self.hidden_dim)
        self.embedding.weight.data.uniform_(-init_range_embed, init_range_embed)
        self.fc.weight.data.uniform_(-init_range_other, init_range_other)
        self.fc.bias.data.zero_()
        for i in range(self.num_layers):
            self.lstm.all_weights[i][0] = torch.FloatTensor(self.embedding_dim, self.hidden_dim).uniform_(-init_range_other, init_range_other)
            self.lstm.all_weights[i][1] = torch.FloatTensor(self.hidden_dim, self.hidden_dim).uniform_(-init_range_other, init_range_other)
    
    def init_hidden(self, batch_size, device):
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        cell = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return hidden, cell
    
    def detach_hidden(self, hidden):
        hidden, cell = hidden
        hidden = hidden.detach()
        cell = cell.detach()
        return hidden, cell

def get_batch(data, seq_len, num_batches, index):
    src = data[:, index: index + seq_len]
    target = data[:, index + 1: index + seq_len + 1]

    return src, target

def trainy(model, data, optimizer, criterion, batch_size, seq_len, clip, device):
    epoch_loss = 0
    model.train()

    num_batches = data.shape[-1] 
    data = data[:, :num_batches - (num_batches - 1) % seq_len]
    num_batches = data.shape[-1] 
    
    hidden = model.init_hidden(batch_size, device)

    for index in range(0, num_batches - 1, seq_len):
        optimizer.zero_grad()
        hidden = model.detach_hidden(hidden)

        src, target = get_batch(data, seq_len, num_batches, index)
        src = src.to(device)
        target = target.to(device)
        batch_size = src.shape[0]
        prediction, hidden = model(src, hidden)

        prediction = prediction.reshape(batch_size * seq_len, -1)
        target = target.reshape(-1)
        loss = criterion(prediction, target)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item() * seq_len

    return epoch_loss / num_batches

def evaluatey(model, data, criterion, batch_size, seq_len, device):
    epoch_loss = 0
    model.eval()

    num_batches = data.shape[-1] 
    data = data[:, :num_batches - (num_batches - 1) % seq_len]
    num_batches = data.shape[-1] 

    hidden = model.init_hidden(batch_size, device)

    with torch.no_grad():
        for index in range(0, num_batches - 1, seq_len):
            hidden = model.detach_hidden(hidden)
            src, target = get_batch(data, seq_len, num_batches, index)
            src = src.to(device)
            target = target.to(device)
            batch_size = src.shape[0]

            prediction, hidden = model(src, hidden)
            prediction = prediction.reshape(batch_size * seq_len, -1)
            target = target.reshape(-1)

            loss = criterion(prediction, target)
            epoch_loss += loss.item() * seq_len

    return epoch_loss / num_batches

def main():
    data = read_file()
    sentences = sent_tokenization(data) # this is a list of sentences in the text 

    clean_sentences = []

    for sentence in sentences:
        clean_sentences.append(tokenization(sentence))
    
    create_vocabulary(clean_sentences)
    words_and_indices(vocabulary)

    batch_size = 64

    clean_sentences.sort()
    random.seed(23)
    random.shuffle(clean_sentences)

    train_size = int(0.7 * len(clean_sentences))
    train = clean_sentences[:train_size]

    dev_size = int(0.15 * len(clean_sentences))
    dev = clean_sentences[train_size:train_size + dev_size]

    test = clean_sentences[train_size + dev_size:]

    train_data = get_data(train, vocabulary, batch_size)
    dev_data = get_data(dev, vocabulary, batch_size)
    test_data = get_data(test, vocabulary, batch_size)

    vocab_size = len(vocabulary)
    embedding_dim = 400
    hidden_dim = 400
    num_layers = 2
    dropout_rate = 0.65
    tie_weights = True
    learning_rate = 0.001

    model = LSTM(vocab_size, embedding_dim, hidden_dim, num_layers, dropout_rate, tie_weights)
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    criterion = nn.CrossEntropyLoss()
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("The model has {} trainable parameters".format(num_params))

    n_epochs = 5
    seq_len = 50
    clip = 0.25
    saved = False

    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.5, patience = 0)

    if saved:
        model.load_state_dict(torch.load('best-val-model.pt', map_location = device))
        test_loss = evaluatey(model, test_data, criterion, batch_size, seq_len, device)
        print("Test Loss: {:.3f}".format(test_loss))
    else:
        best_val_loss = float('inf')

        for epoch in range(n_epochs):
            train_loss = trainy(model, train_data, optimizer, criterion, batch_size, seq_len, clip, device)
            valid_loss = evaluatey(model, dev_data, criterion, batch_size, seq_len, device)

            lr_scheduler.step(valid_loss)

            if valid_loss < best_val_loss:
                best_val_loss = valid_loss
                torch.save(model.state_dict(), 'best-val-model.pt')

            print(f'\tTrain Perplexity: {math.exp(train_loss):.3f}')
            print(f'\tValid Perplexity: {math.exp(valid_loss):.3f}')

if __name__ == "__main__":
    main()