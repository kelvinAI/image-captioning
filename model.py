import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.optim import Adam
import math

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        # Loaded the pre trained resnet model. can use 50 or 18(faster)
        resnet = models.resnet50(pretrained=True)
#         resnet = models.mobilenet_v2(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        # Taking all the front layers except the last
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        
        # Create an  linear layer of size: First dimension of the dropped fully connected layer, and embedding size, so it can be fed into the next RNN embedding layer
        # Same with list(resnet.children())[-1]
        self.embed = nn.Linear(resnet.fc.in_features, embed_size) # use this for resnet
#         self.embed = nn.Linear(list(resnet.children())[-1][-1].in_features, embed_size)  # Use this for mobilenet
        
        

    def forward(self, images):
        features = self.resnet(images)
        # Flatten to fully connected layer
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, drop_prob=0.2):
        super(DecoderRNN, self).__init__()
#         self.hidden_size = hidden_size
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers = num_layers, batch_first=True, dropout=drop_prob)
        self.dropout = nn.Dropout(drop_prob) # Add a dropout layer
        self.scores = nn.Linear(self.lstm.hidden_size, vocab_size) 
        
        #Initilize weights
        self.init_weights()
    
    def init_weights(self):
        '''
        Initialize the weights for the word embedding layer and the final FC layer
        '''
#         torch.nn.init.xavier_uniform_(self.scores.weight)
#         torch.nn.init.xavier_uniform_(self.word_embeddings.weight)

        # Set bias tensor to all 0.01
        self.scores.bias.data.fill_(0.01)
        # FC weights as xavier normal
        torch.nn.init.xavier_normal_(self.scores.weight)

        # init forget gate bias to 1
        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n,  names):
                bias = getattr(self.lstm, name)
                n = bias.size(0)
                start, end = n//4, n//2
                bias.data[start:end].fill_(1.)

    
    def forward(self, features, captions):
        embeddings = self.word_embeddings(captions[:,:-1]) # Discarding the last token <end>  because we will add the features vector as the first step instead
        # Now add the feature vector  as the first step of the training,
        # The current shape for word_embeddings is 10, 12 , 256, we will transform it to (10, 13, 256) 
        # input features from cnn is in the shape of 10, 256 (need to add one dimension by unsqueeze at the 1st dimension
        # features.unsqueez(1) creates a tensor of shape (10, 1, 256) , which can be concatenated with (10, 12, 256) at dim = 1
        # The way to add the features from CNN into the embeddings as the first step is by concating the matrix
        lstm_input = torch.cat([features.unsqueeze(1), embeddings],dim=1)
        # Now lstm_input is of shape (batch_size, 13, 256 )
        
        out, (hidden, cell) = self.lstm(lstm_input)
        # Output refers to the hidden state of the last RNN, which is equivalent to hidden[:,-1 ,:] (batch, layers, hidden_nodes)
        # Add a dropout layer
        out = self.dropout(out)
        output = self.scores(out)
#         output.view(seq_len, batch, num_directions, hidden_size)
#         print(f'Shapes: out:{out.shape}, output:{output.shape} hidden:{hidden.shape} cell:{cell.shape}')
        return output
        
    

    def predict(self, inputs, states=None, max_len=20):
        # Not using word embeddings layer as there are no text input
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
#         print("Decoder sample")
        output_tokens = []
        _input = inputs
        for i in range(max_len):
#             print(f"Lstm Input shape:{_input.shape}")  # [1, 1, 256]
            
            out, states = self.lstm(_input, states)
#             print(f"out after lstm:{out.shape} ") # [1, 1, 256]
            out = self.scores(out.squeeze(1))
#             print(f"out after scores:{out.shape} ") # [1, 10330]
            
            
#             tensor_list = sorted([ (i, t.item()) for i, t in enumerate(out[0,:])], key=lambda x:x[1], reverse=True)
#             print(tensor_list[:5])
            token = out.argmax(dim=1)
            output_tokens.append(token.item())
            token_reshaped = token.unsqueeze(0)
#             print(f"token after argmax:{token.shape}, token appended{token.item()}, token reshaped shape:{token_reshaped.shape}, token reshaped into word embedding:{token_reshaped}")
            _input = self.word_embeddings(token_reshaped)
#             _input = _input.unsqueeze(0)
#             print(f"_input shape after word embedding layer{_input.shape}")
            
            
        
        return output_tokens

