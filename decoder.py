import torch
import torch.nn as nn
import torch.nn.functional as F

from autoencoder import Autoencoder
from vocabulary import Vectorizer


class Decoder(nn.Module):
    def __init__(self, embed_size: int, hidden_size: int, vocab_size: int, num_layers: int, vocab: Vectorizer):
        super(Decoder, self).__init__()
        self.encoder = Autoencoder()
        self.encoder = self.encoder.encoder

        for param in list(self.encoder.parameters())[:-1]:
            param.requires_grad = False

        self.embed = nn.Embedding(
            vocab_size, embed_size, padding_idx=vocab.padding_idx)
        self.lstm = nn.LSTM(embed_size, hidden_size,
                            num_layers, batch_first=True)
        self.linear = nn.Sequential(
            nn.Linear(hidden_size, hidden_size*2),
            nn.Linear(hidden_size*2, vocab_size),

        )
        self.dropout = nn.Dropout(0.5)

    def forward(self, img, translations):
        features = -self.encoder(img)
        embeddings = self.dropout(self.embed(translations))
        x = torch.cat((features.unsqueeze(dim=1), embeddings), dim=1)
        hiddens, _ = self.lstm(x)
        outputs = self.linear(hiddens)
        return outputs

    def caption_image(self, features, vocabulary, max_length=50):
        result_caption = []
        features = -self.encoder(features.unsqueeze(0)).squeeze(0)
        with torch.no_grad():
            states = None
            for _ in range(max_length):
                hiddens, states = self.lstm(features.unsqueeze(0), states)
                output = self.linear(hiddens.squeeze(dim=0))
                predicted = output.argmax(dim=0)
                result_caption.append(predicted.item())

                features = self.embed(predicted)

                if vocabulary.end_of_sentance_idx == predicted.item():
                    break
        print(result_caption)
        return vocabulary.decode(result_caption)


class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()

        self.attention_dim = attention_dim

        self.W = nn.Linear(decoder_dim, attention_dim)
        self.U = nn.Linear(encoder_dim, attention_dim)

        self.A = nn.Linear(attention_dim, 1)

    def forward(self, features, hidden_state):

        u_hs = self.U(features)
        w_ah = self.W(hidden_state)

        combined_states = torch.tanh(u_hs + w_ah.unsqueeze(1))

        attention_scores = self.A(combined_states)
        attention_scores = attention_scores.squeeze(2)

        alpha = F.softmax(attention_scores, dim=1)

        attention_weights = features * alpha.unsqueeze(2)
        attention_weights = attention_weights.sum(dim=1)

        return alpha, attention_weights


class AttentionDecoder(nn.Module):
    def __init__(self, embed_size: int, hidden_size: int, encoder_size: int, attention_dim: int, vocab: Vectorizer):
        super(AttentionDecoder, self).__init__()

        self.encoder = nn.Sequential(
            *list(Autoencoder().encoder.children())[:-2])

        for param in list(self.encoder.parameters())[:-1]:
            param.requires_grad = False

        self.vocab = vocab
        self.embedding = nn.Embedding(
            len(self.vocab), embed_size, padding_idx=self.vocab.padding_idx)
        self.pos_embeddings = nn.Embedding(self.vocab.max_len, embed_size)
        self.attention = Attention(encoder_size, hidden_size, attention_dim)

        self.init_h = nn.Linear(encoder_size, hidden_size)
        self.init_c = nn.Linear(encoder_size, hidden_size)

        self.lstm_cell = nn.LSTMCell(
            embed_size+encoder_size, hidden_size, bias=True)
        self.f_beta = nn.Linear(hidden_size, encoder_size)

        self.fcn = nn.Linear(hidden_size, len(self.vocab))
        self.drop = nn.Dropout(0.3)

    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward_step(self, decoder_input, encoder_outputs, last_hidden, last_cell, position, device='cuda:0'):

        embeds = self.embedding(
            decoder_input) + self.pos_embeddings(torch.tensor(position).long().to(device))
        alpha, context = self.attention(encoder_outputs, last_hidden)

        lstm_input = torch.cat((embeds, context), dim=1)

        hidden, cell = self.lstm_cell(lstm_input, (last_hidden, last_cell))

        output = self.fcn(self.drop(hidden))

        return output, alpha, hidden, cell

    def forward(self, imgs, decoder_input, device='cuda:0'):

        encoder_outputs = self.encoder(imgs)
        encoder_outputs = encoder_outputs.permute(0, 2, 3, 1)
        encoder_outputs = encoder_outputs.view(
            encoder_outputs.size(0), -1, encoder_outputs.size(3))

        hidden, cell = self.init_hidden_state(encoder_outputs)

        seq_length = len(decoder_input[0])
        batch_size = decoder_input.size(0)
        num_features = encoder_outputs.size(1)

        outputs = torch.zeros(batch_size, seq_length,
                              len(self.vocab)).to(device)
        alphas = torch.zeros(batch_size, seq_length, num_features).to(device)

        for s in range(seq_length):
            output, alpha, hidden, cell = self.forward_step(
                decoder_input[:, s].to(device), encoder_outputs, hidden, cell, s)
            outputs[:, s] = output
            alphas[:, s] = alpha

        return outputs, alphas

    def greedy_decode(self, imgs, device='cpu'):

        encoder_outputs = self.encoder.to(device)(imgs)
        encoder_outputs = encoder_outputs.permute(0, 2, 3, 1)
        encoder_outputs = encoder_outputs.view(
            encoder_outputs.size(0), -1, encoder_outputs.size(3))

        batch_size = encoder_outputs.size(0)
        hidden, cell = self.init_hidden_state(encoder_outputs)
        decoder_input = torch.tensor(self.vocab.text2seq['<SOS>']).to(device)
        decoder_input = torch.LongTensor([decoder_input]).to(device)

        decoded_batch = [self.vocab.text2seq['<SOS>']]
        for i in range(self.vocab.max_len):
            decoder_output, alpha, hidden, cell = self.forward_step(
                decoder_input, encoder_outputs, hidden, cell, i, 'cpu')

            decoder_output = decoder_output.view(batch_size, -1)
            predicted_word_idx = decoder_output.argmax(dim=1)
            decoded_batch.append(predicted_word_idx.item())
            if self.vocab.seq2text[predicted_word_idx.item()] == "<EOS>":
                break
            decoder_input = predicted_word_idx
        return decoded_batch
