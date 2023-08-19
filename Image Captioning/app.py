from flask import Flask, request, jsonify, send_from_directory, render_template
from torchvision import transforms
from PIL import Image
import torch
from build_vocab import Vocabulary
import build_vocab
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
import torch.nn as nn
import torchvision.models as models
import nltk
from collections import Counter


app = Flask(__name__)

# Load your model here
encoder_path  = './model_weights/encoder_v2.ckpt'
decoder_path  = './model_weights/decoder_v2.ckpt'

vocabs = build_vocab.build_vocab('./uitviic_captions_train2017.json', 2)


transform = transforms.Compose([ 
     transforms.CenterCrop(224),
     transforms.ToTensor(), 
     transforms.Normalize((0.485, 0.456, 0.406), 
                          (0.229, 0.224, 0.225))
 ])


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        
    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features
    
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length
        
    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs
    
    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)                        # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder_model  = EncoderCNN(300).to(device).eval()
decoder_model  = DecoderRNN(300, 512, len(vocabs), 1).to(device)

encoder_model.load_state_dict(torch.load(encoder_path, map_location=device))
decoder_model .load_state_dict(torch.load(decoder_path, map_location=device ))


@app.route('/upload', methods=['GET', 'POST'])
def predict():
    image_file = request.files['image']
    image = Image.open(image_file).convert('RGB')
    image = image.resize((224, 224), Image.ANTIALIAS)
    image = transform(image).unsqueeze(0)
    image_tensor = image.to(device)


    features = encoder_model(image_tensor)
    sampled_ids = decoder_model.sample(features)
    sampled_ids = sampled_ids[0].cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)

    sampled_caption = []
    for word_id in sampled_ids:
        word = vocabs.idx2word[word_id]
        if word not in {'<start>', '<end>'}:  # Skip these tokens
            sampled_caption.append(word)
        if word == '<end>':
            break
    sentence = ' '.join(sampled_caption)
    caption = 'This is the caption for the image'

    return jsonify({'caption': sentence})

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return '', 204

if __name__ == '__main__':
    app.run(debug=True)

