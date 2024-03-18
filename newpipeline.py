import nltk
from collections import Counter
import argparse
import os
import json
import multiprocessing
import torch.nn as nn
import torch.nn.init
import torch.nn.functional as F
import torch.optim as optim

annotations = {
    'rsicd_precomp': ['train_caps.txt', 'val_caps.txt', 'test_caps.txt'],
    'rsitmd_precomp': ['train_caps.txt', 'val_caps.txt'],
    }


class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 1

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


def serialize_vocab(vocab, dest):
    d = {}
    d['word2idx'] = vocab.word2idx
    d['idx2word'] = vocab.idx2word
    d['idx'] = vocab.idx
    with open(dest, "w") as f:
        json.dump(d, f)


def deserialize_vocab(src):
    with open(src) as f:
        d = json.load(f)
    vocab = Vocabulary()
    vocab.word2idx = d['word2idx']
    vocab.idx2word = d['idx2word']
    vocab.idx = d['idx']
    return vocab


def from_txt(txt):
    captions = []
    with open(txt, 'rb') as f:
        for line in f:
            captions.append(line.strip())
    return captions


def build_vocab(data_path, data_name, caption_file, threshold):
    """Build a simple vocabulary wrapper."""

    stopword_list = list(set(nltk.corpus.stopwords.words('english')))
    counter = Counter()
    for path in caption_file[data_name]:
        full_path = os.path.join(os.path.join(data_path, data_name), path)
        captions = from_txt(full_path)

        for i, caption in enumerate(captions):
            tokens = nltk.tokenize.word_tokenize(
                caption.lower().decode('utf-8'))
            punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']
            tokens = [k for k in tokens if k not in punctuations]
            tokens = [k for k in tokens if k not in stopword_list]
            counter.update(tokens)

            if i % 1000 == 0:
                print("[%d/%d] tokenized the captions." % (i, len(captions)))

    # Discard if the occurrence of the word is less than min_word_cnt.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()

    # Add words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    vocab.add_word('<unk>')

    return vocab


def main(data_path, data_name):
    vocab = build_vocab(data_path, data_name, caption_file=annotations, threshold=5)
    serialize_vocab(vocab, 'vocab/%s_vocab.json' % data_name)
    print("Saved vocabulary file to ", 'vocab/%s_vocab.json' %(data_name))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='data')
    parser.add_argument('--data_name', default='sydney_precomp',
                        help='{coco,f30k}')
    opt = parser.parse_args()
    main('/Users/radhikagupta/Downloads/SOP', 'rsicd_precomp')


##############################################################################################
#DATA PREPROCESSING 
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import nltk
import numpy as np
import argparse
from PIL import Image

class PrecompDataset(data.Dataset):
    """
    Load precomputed captions and image features
    """

    def __init__(self, data_split, vocab, opt):
        self.vocab = vocab
        self.loc = '/Users/radhikagupta/Downloads/SOP/rsicd_precomp/'
        self.img_path = '/Users/radhikagupta/Downloads/SOP/RSICD_images/'
        
        # Captions
        self.captions = []
        self.maxlength = 0

        # local features
        local_features = np.load('/Users/radhikagupta/Downloads/SOP/rsicd_local.npy', allow_pickle=True)[()]

        if data_split != 'test':
            with open(self.loc+'%s_caps_verify.txt' % data_split, 'rb') as f:
                for line in f:
                    self.captions.append(line.strip())

            self.images = []
            self.local_adj = []
            self.local_rep = []
            with open(self.loc + '%s_filename_verify.txt' % data_split, 'rb') as f:
                for line in f:
                    # local append
                    filename = str(line.strip())[2:-1].split(".")[0] + ".txt"
                    self.local_adj.append(np.array(local_features['adj_matrix'][filename]))
                    self.local_rep.append(np.array(local_features['local_rep'][filename]))

                    self.images.append(line.strip())
        else:
            with open(self.loc + '%s_caps.txt' % data_split, 'rb') as f:
                for line in f:
                    self.captions.append(line.strip())

            self.images = []
            self.local_adj = []
            self.local_rep = []
            with open(self.loc + '%s_filename.txt' % data_split, 'rb') as f:
                for line in f:
                    # local append
                    filename = str(line.strip())[2:-1].split(".")[0] + ".txt"
                    self.local_adj.append(np.array(local_features['adj_matrix'][filename]))
                    self.local_rep.append(np.array(local_features['local_rep'][filename]))

                    self.images.append(line.strip())

        self.length = len(self.captions)
        #If data has redundancy in images, we divide by 5,
        if len(self.images) != self.length:
            self.im_div = 5
        else:
            self.im_div = 1

        if data_split == "train":
            self.transform = transforms.Compose([
                transforms.Resize((278, 278)),
                transforms.RandomRotation(degrees=(0, 90)),
                transforms.RandomCrop(256),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])

    def __getitem__(self, index):
        # handle the image redundancy
        img_id = index//self.im_div
        caption = self.captions[index]

        vocab = self.vocab

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(
            caption.lower().decode('utf-8'))
        punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']
        tokens = [k for k in tokens if k not in punctuations]
        tokens_UNK = [k if k in vocab.word2idx.keys() else '<unk>' for k in tokens]


        caption = []
        caption.extend([vocab(token) for token in tokens_UNK if token!='<unk>'])
        caption = torch.LongTensor(caption)

        image = Image.open(self.img_path +'/' +str(self.images[img_id])[2:-1]).convert('RGB')
        image = self.transform(image)  # torch.Size([3, 256, 256])

        # local
        local_rep =  torch.from_numpy(self.local_rep[img_id]).type(torch.float32)
        local_adj = torch.from_numpy(self.local_adj[img_id]).type(torch.float32)


        return image, caption, tokens_UNK, index, img_id, local_rep, local_adj

    def __len__(self):
        return self.length


def collate_fn(data):

    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[4]) if isinstance(x[4], list) else 0, reverse=True)
    images, captions, tokens, ids, img_ids,local_rep,local_adj = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)

    local_rep = torch.stack(local_rep, 0)
    local_adj = torch.stack(local_adj, 0)

    # Merge captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    lengths = [l if l !=0 else 1 for l in lengths]

    return images, targets, lengths, ids, local_rep, local_adj


def get_precomp_loader(data_split, vocab, batch_size=100,
                       shuffle=True, num_workers=0, opt={}):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    dset = PrecompDataset(data_split, vocab, opt)

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=False,
                                              collate_fn=collate_fn,
                                              num_workers=num_workers)
    return data_loader

def get_loaders(vocab,opt):
    train_loader = get_precomp_loader( 'train', vocab, 100, True, 0,opt)
    val_loader = get_precomp_loader( 'val', vocab, 70, False, 0,opt)
    return train_loader, val_loader


def get_test_loader(vocab, opt):
    test_loader = get_precomp_loader( 'test', vocab,
                                      70, False, 2,opt)
    return test_loader


#################################################################
#Creating A Model 

from torchvision import models, transforms
from vit_pytorch import ViT


# Define ViT model
class ViTFeatureExtractor(nn.Module):
    def __init__(self, image_size=256, patch_size=32, num_classes=1000, dim=768, depth=12, heads=12, mlp_dim=3072):
        super(ViTFeatureExtractor, self).__init__()
        self.transformer = ViT(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=num_classes,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            pool='cls',
            channels=3  # Assuming RGB images
        )

    def forward(self, x):
        return self.transformer(x)

# Initialize ViT model
vit_model = ViTFeatureExtractor()
print("Checkpoint 1 ******************************************:")
vocab = Vocabulary()
vocab_path = '/Users/radhikagupta/Downloads/SOP/rsitmd_splits_vocab.json'  # Update with the correct path
vocab = deserialize_vocab(vocab_path)

#Importing BERT Model 
from transformers import BertTokenizer, BertModel
# Load BERT tokenizer
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
opt = {}

#Initialising BERT Model
bert_model = BertModel.from_pretrained('bert-base-uncased')
print("Checkpoint 2 ******************************************:")
def extract_bert_features(bert_inputs):
    
    # Forward pass through BERT model
    with torch.no_grad():
        outputs = bert_model(**bert_inputs)

    # Extract the output embeddings from BERT
    embeddings = outputs.last_hidden_state

    # You can use the pooled output or any other representation based on your task
    pooled_output = outputs.pooler_output

    return pooled_output


# Use get_loaders to get train_loader and val_loader  
train_loader, val_loader = get_loaders(vocab, opt)
print("Checkpoint 3 ******************************************:")
num_images_to_process = 10
count = 0
for batch in train_loader:
    images, captions, _, img_ids,_,_ = batch
    # Set models to evaluation mode
    vit_model.eval()
    bert_model.eval()

    # ViT feature extraction
    with torch.no_grad():
        vit_features = vit_model(images)

    # BERT feature extraction
    captions_list = []
    for cap in captions:
        
        tokens = [vocab.idx2word[str(idx.item())] for idx in cap if idx.item() != 0]
        caption_str = ' '.join(tokens)
        captions_list.append(caption_str)

    bert_inputs = bert_tokenizer(captions_list, return_tensors='pt', padding=True, truncation=True, max_length=128)
    
    with torch.no_grad():
        print("Checkpoint 4 ******************************************:")
        bert_features = extract_bert_features(bert_inputs)
        print("Checkpoint 5 ******************************************:")
        
    count += len(images)
    if count >= num_images_to_process:
        break
# Now 'vit_features' contains the ViT embeddings for images
# and 'bert_features' contains the BERT embeddings for captions

print("Image Features Shape:", vit_features.shape)
print("Caption Features Shape:", bert_features.shape)

#Defining a Cross Attention Model (code used from the AMFMN paper)

class CrossAttention(nn.Module):

    def __init__(self, opt={}):
        super(CrossAttention, self).__init__()

        self.att_type = "soft_att"
        dim = 512

        if self.att_type == "soft_att":
            self.cross_attention = nn.Sequential(
                nn.Linear(dim, dim),
                nn.Sigmoid()
            )
        elif self.att_type == "fusion_att":
            self.cross_attention_fc1 = nn.Sequential(
                nn.Linear(2*dim, dim),
                nn.Sigmoid()
            )
            self.cross_attention_fc2 = nn.Sequential(
                nn.Linear(2*dim, dim),
            )
            self.cross_attention = lambda x:self.cross_attention_fc1(x)*self.cross_attention_fc2(x)

        elif self.att_type == "similarity_att":
            self.fc_visual = nn.Sequential(
                nn.Linear(dim, dim),
            )
            self.fc_text = nn.Sequential(
                nn.Linear(dim, dim),
            )
        else:
            raise Exception

    def forward(self, visual, text):
        batch_v = visual.shape[0]
        batch_t = text.shape[0]

        if self.att_type == "soft_att":
            visual_gate = self.cross_attention(visual)

            # mm
            visual_gate = visual_gate.unsqueeze(dim=1).expand(-1, batch_t, -1)
            text = text.unsqueeze(dim=0).expand(batch_v, -1, -1)

            return visual_gate*text

        elif self.att_type == "fusion_att":
            visual = visual.unsqueeze(dim=1).expand(-1, batch_t, -1)
            text = text.unsqueeze(dim=0).expand(batch_v, -1, -1)

            fusion_vec = torch.cat([visual,text], dim=-1)

            return self.cross_attention(fusion_vec)
        elif self.att_type == "similarity_att":
            visual = self.fc_visual(visual)
            text = self.fc_text(text)

            visual = visual.unsqueeze(dim=1).expand(-1, batch_t, -1)
            text = text.unsqueeze(dim=0).expand(batch_v, -1, -1)

            sims = visual*text
            return F.sigmoid(sims) * text
        
#Initialising the cross attention model 
cross_attention_model = CrossAttention()


# Defining a simple feedforward neural network (FFN)
class FFN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FFN, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)

#Reshaping the vit and BERT features using the FFN 

vit_ffn = FFN(vit_features.shape[1], 512)
vit_features_reshaped = vit_ffn(vit_features)

bert_ffn = FFN(bert_features.shape[1], 512)
bert_features_reshaped = bert_ffn(bert_features)

print("AFTER RESHAPING ______________:")

print("Image Features Shape:", vit_features_reshaped.shape)
print("Caption Features Shape:", bert_features_reshaped.shape)

import torch.nn.functional as F

# Apply L2 normalization to VIT and BERT features, The p=2 argument specifies L2 normalization.
vit_features_normalized = F.normalize(vit_features_reshaped, p=2, dim=1)
bert_features_normalized = F.normalize(bert_features_reshaped, p=2, dim=1)

# Apply cross-attention to vit_features
vit_features_with_attention = cross_attention_model(vit_features_reshaped, bert_features_reshaped)

# Apply cross-attention to bert_features
bert_features_with_attention = cross_attention_model(bert_features_reshaped, vit_features_reshaped)

# def l2norm(X, dim, eps=1e-8):
#     """L2-normalize columns of X
#     """
#     norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
#     X = torch.div(X, norm)
#     return X

import torch
import torch.nn as nn
import torch.nn.functional as F

class GCN(nn.Module):
    def __init__(self , dim_in=20 , dim_out=20, dim_embed = 512):
        super(GCN,self).__init__()

        self.fc1 = nn.Linear(dim_in ,dim_in,bias=False)
        self.fc2 = nn.Linear(dim_in,dim_in//2,bias=False)
        self.fc3 = nn.Linear(dim_in//2,dim_out,bias=False)

        self.out = nn.Linear(dim_out * dim_in, dim_embed)

    def forward(self, A, X):
        batch, objects, rep = X.shape[0], X.shape[1], X.shape[2]

        # first layer
        tmp = (A.bmm(X)).view(-1, rep)
        X = F.relu(self.fc1(tmp))
        X = X.view(batch, -1, X.shape[-1])

        # second layer
        tmp = (A.bmm(X)).view(-1, X.shape[-1])
        X = F.relu(self.fc2(tmp))
        X = X.view(batch, -1, X.shape[-1])

        # third layer
        tmp = (A.bmm(X)).view(-1, X.shape[-1])
        X = F.relu(self.fc3(tmp))
        X = X.view(batch, -1)

        # Output layer
        X = self.out(X)

        return F.normalize(X, p=2, dim=-1)  # L2 normalization of the output

count = 0; 
for batch in train_loader:
    images, _, _, _,local_rep, local_adj = batch
    # Instantiate GCN model
    gcn_model = GCN(dim_in=20, dim_out=20, dim_embed=512)
    print("Checkpoint 6 ******************************************:")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(gcn_model.parameters(), lr=0.01)
    num_epochs = 1

    print("Checkpoint 7 ******************************************:")

    gcn_features = gcn_model(local_rep, local_adj)
    count += len(images)
    if count >= num_images_to_process:
        break
    
print("Local Features(After GCN) shape", gcn_features.shape)
