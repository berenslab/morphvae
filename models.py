## This file holds all the models I have created so far
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from numpy import random
from collections import OrderedDict

        
        
class SeqEncoder(nn.Module):
    def __init__(self,input_dim: int, embed_dim: int, hid_dim: int, n_layers=2, dropout=.5 ):
        super().__init__()
        
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.n_dir = 1
        self.embed_dim = embed_dim
        self.embedding = nn.Linear(input_dim,embed_dim)
        self.rnn = nn.LSTM(embed_dim, hid_dim, n_layers, dropout = dropout)
        
        self.dropout = nn.Dropout(dropout)
      
    
     
    def forward(self, src, src_len):
        
        #src = [src len, batch size, input dim]
        
        embedded = self.dropout(self.embedding(src))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, src_len, enforce_sorted=False)
        
        #embedded = [src len, batch size, input dim]
        
        packed_outputs, (hidden, cell) = self.rnn(packed_embedded)
        
    
        #outputs = [src len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #outputs are always from the top hidden layer
                
        return hidden, cell 

    
    
class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)
        
    def forward(self, hidden, encoder_outputs, mask):
        
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [src len, batch size, enc hid dim * 2]
        
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        
        #repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
  
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        #hidden = [batch size, src len, dec hid dim]
        #encoder_outputs = [batch size, src len, enc hid dim * 2]
        
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2))) 
        
        #energy = [batch size, src len, dec hid dim]

        attention = self.v(energy).squeeze(2)
        
        #attention = [batch size, src len]
        
        attention = attention.masked_fill(mask == 0, -1e10)
        
        return F.softmax(attention, dim = 1)
    
class SeqDecoder(nn.Module):
    def __init__(self, output_dim, embed_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embed_dim = embed_dim
        
        self.embedding = nn.Linear(output_dim, embed_dim)
        
        self.rnn = nn.LSTM(embed_dim, hid_dim, n_layers, dropout = dropout)
        
        self.fc_out = nn.Linear(hid_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)

        
    def forward(self, input, hidden, cell):
           
        #input = [batch size, out_dim]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #n directions in the decoder will both always be 1, therefore:
        #hidden = [n layers, batch size, hid dim]
        #context = [n layers, batch size, hid dim]
        
        input = input.unsqueeze(0)
        
        #input = [1, batch size,out dim]
        
        embedded = self.dropout(self.embedding(input))
        
        #embedded = [1, batch size, embed dim]
                
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        
        #output = [seq len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #seq len and n directions will always be 1 in the decoder, therefore:
        #output = [1, batch size, hid dim]
        #hidden = [n layers, batch size, hid dim]
        #cell = [n layers, batch size, hid dim]
        
        prediction = self.fc_out(output.squeeze(0))
        
        #prediction = [batch size, output dim]
        
        return prediction, hidden, cell
    

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
    
    
    
    def forward(self, src,src_len, trg, teacher_forcing_ratio = 0.5):
        
        #src = [src len, batch size, input dim]
        #trg = [trg len, batch size, input dim]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src,src_len)

        #first input to the decoder is the last coordinate
        input = trg[0,:]
        
        for t in range(1, trg_len):
            
            #insert input token embedding, previous hidden and previous cell states
            #receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input, hidden, cell)
           
        
            #place predictions in a tensor holding predictions for each token
            outputs[t] = output
            
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            
            #get the highest predicted token from our predictions
            top1 = output
            
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            input = trg[t] if teacher_force else top1
        
        return outputs
    
    
class Seq2SeqDataSet(torch.utils.data.Dataset):
    
    def __init__(self, data, labels, max_length, n_walks, output_dim, masking_element, transform=None):
        self.data = data
        self.labels = labels
        self.max_length = max_length
        self.n_walks = n_walks
        self.output_dim = output_dim
        self.masking_el = masking_element
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        
        padded_trg = torch.ones((self.n_walks, self.max_length, self.output_dim)) *self.masking_el
        padded_src = torch.ones((self.n_walks, self.max_length, self.output_dim)) *self.masking_el
        seq_lengths = torch.zeros((self.n_walks,))
        for k, trg in enumerate(self.data[index]):
            if self.transform is not None:
                trg = self.transform(trg)
            src = trg.flip(dims=[0])
            padded_trg[k,:len(trg),:] = trg
            padded_src[k,:len(src),:] = src
            seq_lengths[k] = len(trg)
        if self.labels is not None:
            return padded_src, padded_trg, seq_lengths, index, self.labels[index]
        else:
            return padded_src, padded_trg, seq_lengths, index
        
class PoolingClassifier(nn.Module):
    
    def __init__(self, hid_dim, num_classes, pooling_size, dropout, pooling='max'):
        """
            hid_dim : int = enc_hid_dim*num_layers
            num_classes: int
            pooling_size: size for 2D max pooling = n_walks
        """
        super().__init__()
        self.hid_dim = hid_dim
        self.num_classes = num_classes
        self.pooling_size = pooling_size
        
        if pooling == 'max':
            self.pooling_layer = nn.MaxPool2d(kernel_size=(pooling_size,1))
        elif pooling == 'avg':
            self.pooling_layer = nn.AvgPool2d(kernel_size=(pooling_size,1))
            
        self.fc = nn.Linear(hid_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,X):
        
        # X = [batch size * n_walks, enc_hid_dim]
        _, enc_hid_dim = X.shape
        
        X = X.reshape(-1,1,self.pooling_size, enc_hid_dim)
        
        # X = [batch size, 1, n_walks, enc hid dim]
        
        # max pool over all RWs in one neuron
        pooled_X = self.pooling_layer(X).squeeze()
        
        output = self.fc(self.dropout(pooled_X))
        return output
    

class Seq2Seq_VAE(nn.Module):
    def __init__(self, encoder, decoder, dist, device):
        """
        encoder:
        decoder:
        dist:
        device: 
        """
        super().__init__()
        
        self.encoder = encoder
        
        self.dist = dist
        self.states_to_latent = nn.Linear(encoder.hid_dim*encoder.n_layers*2, dist.lat_dim)
        self.latent_to_states = nn.Linear(dist.lat_dim, decoder.hid_dim*decoder.n_layers*2)
        self.decoder = decoder
        self.device = device
        
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
    
    
    def encode(self, src, src_len):
        
        hidden, cell = self.encoder(src,src_len)

        n_layers, batch_size, hid_dim = hidden.shape
        # hidden, cell = [n_layers, batch size, hid dim]
        # project to latent space
        states = torch.cat((hidden,cell), dim=0).permute(1,0,2).reshape(batch_size, -1)
    
        # states = [batch size, n_layers *2 * hid dim]
        self.h = self.states_to_latent(states)
        
        # h = [batch size, latent dim]
        
        # sample from vMF distribution
        tup, kld, vecs = self.dist.build_bow_rep(self.h, n_sample=5)

        self.Z = torch.mean(vecs, dim=0)
        decoder_states = self._get_decoder_states(self.Z, batch_size)
    
        return decoder_states
        
        
    def _get_decoder_states(self, latent, batch_size):
        
        decoder_states = self.latent_to_states(latent).reshape(batch_size, -1, 2)
        # decoder_states = [batch size, hid_dim * n layers, 2]
        
        hidden = decoder_states[:,:,0].reshape(batch_size,self.decoder.hid_dim, self.decoder.n_layers).permute(2,0,1).contiguous()
        cell = decoder_states[:,:,1].reshape(batch_size,self.decoder.hid_dim, self.decoder.n_layers).permute(2,0,1).contiguous()
        
        return hidden, cell
    
    def forward(self, src,src_len, trg, teacher_forcing_ratio = 0.5):
        
        #src = [src len, batch size, input dim]
        #trg = [trg len, batch size, input dim]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        #last hidden states of the encoder is used as the initial hidden states of the decoder
        
        hidden, cell = self.encode(src,src_len)
        
        #first input to the decoder is the first coordinate
        input = trg[0,:]
        
        for t in range(1, trg_len):
            
            #insert input token embedding, previous hidden and previous cell states
            #receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input, hidden, cell)
           
        
            #place predictions in a tensor holding predictions for each token
            outputs[t] = output
            
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            
            #get the predictions
            top1 = output
            
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            input = trg[t] if teacher_force else top1

        return outputs
    
def init_weights(m):
    #for name, param in m.named_parameters():
    #    nn.init.uniform_(param.data, -0.08, 0.08)  
        
    for p in m.parameters():
        if p.data.ndimension() >= 2:
            torch.nn.init.xavier_uniform_(p.data)
        else:
            torch.nn.init.zeros_(p.data)