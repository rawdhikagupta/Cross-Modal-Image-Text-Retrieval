import nltk
from collections import Counter
import argparse
import os
import json
import multiprocessing

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

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import nltk
import numpy as np
import yaml
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
        #local_features = np.load('/Users/radhikagupta/Downloads/SOP/rsicd_local.npy', allow_pickle=True)

        if data_split != 'test':
            with open(self.loc+'%s_caps_verify.txt' % data_split, 'rb') as f:
                for line in f:
                    self.captions.append(line.strip())

            self.images = []
            #self.local_adj = []
            #self.local_rep = []
            with open(self.loc + '%s_filename_verify.txt' % data_split, 'rb') as f:
                for line in f:
                    # local append
                    filename = str(line.strip())[2:-1].split(".")[0] + ".txt"
                    #self.local_adj.append(np.array(local_features['adj_matrix'][filename]))
                    #self.local_rep.append(np.array(local_features['local_rep'][filename]))

                    self.images.append(line.strip())
        else:
            with open(self.loc + '%s_caps.txt' % data_split, 'rb') as f:
                for line in f:
                    self.captions.append(line.strip())

            self.images = []
            #self.local_adj = []
            #self.local_rep = []
            with open(self.loc + '%s_filename.txt' % data_split, 'rb') as f:
                for line in f:
                    # local append
                    filename = str(line.strip())[2:-1].split(".")[0] + ".txt"
                    #self.local_adj.append(np.array(local_features['adj_matrix'][filename]))
                    #self.local_rep.append(np.array(local_features['local_rep'][filename]))

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
        #local_rep =  torch.from_numpy(self.local_rep[img_id]).type(torch.float32)
        #local_adj = torch.from_numpy(self.local_adj[img_id]).type(torch.float32)


        return image, caption, tokens_UNK, index, img_id

    def __len__(self):
        return self.length


def collate_fn(data):

    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[4]) if isinstance(x[4], list) else 0, reverse=True)
    images, captions, tokens, ids, img_ids = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)

    #local_rep = torch.stack(local_rep, 0)
    #local_adj = torch.stack(local_adj, 0)

    # Merge captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    lengths = [l if l !=0 else 1 for l in lengths]

    return images, targets, lengths, ids


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


import torch
import torch.nn as nn
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

vocab = Vocabulary()
vocab_path = '/Users/radhikagupta/Downloads/SOP/rsitmd_splits_vocab.json'  # Update with the correct path
vocab = deserialize_vocab(vocab_path)

opt = {}  
train_loader, val_loader = get_loaders(vocab, opt)

def extract_vit_features(data_loader, model, num_images=None):
    model.eval()
    all_features = []
    image_ids = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            images, _, _,img_ids = batch  # assuming the image IDs are present in the last element of the batch tuple
            features = model(images)
            all_features.append(features)
            image_ids.extend(img_ids)

            # Check if the specified number of images has been processed
            if num_images is not None and len(all_features) * images.size(0) >= num_images:
                remaining_images = num_images - (len(all_features) - 1) * images.size(0)
                all_features[-1] = all_features[-1][:remaining_images]
                image_ids = image_ids[:num_images]
                break  # Stop processing more batches

    return torch.cat(all_features, dim=0), image_ids

# Set the number of images you want to process
num_images_to_process = 1000

# Extract features from the training loader for the specified number of images

#Calling the extract_vit_features function
train_features_subset, image_ids_subset = extract_vit_features(train_loader, vit_model, num_images=num_images_to_process)

# Now 'train_features_subset' contains the extracted features for the subset of images
# and 'image_ids_subset' contains the corresponding image IDs
print("Shape of extracted features:", train_features_subset.shape)
print("Number of image IDs:", len(image_ids_subset))


############################################################################
import torch
from transformers import BertTokenizer, BertModel

# Load BERT tokenizer
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Use get_loaders to get train_loader and val_loader
vocab = Vocabulary()
opt = {}  # You can customize the options if needed
train_loader, val_loader = get_loaders(vocab, opt)

# Extract captions from the train_loader
captions_list = []
num_sentences_to_process = 1000
for batch in train_loader:
    _, captions, _, _ = batch
    for cap in captions:
        tokens = [vocab.idx2word[str(idx.item())] for idx in cap]
        caption_str = ' '.join(tokens)
        captions_list.append(caption_str)
        
        if len(captions_list) >= num_sentences_to_process:
            break

        # Check if the specified number of sentences has been processed
        if len(captions_list) >= num_sentences_to_process:
            break


# Tokenize captions using BERT tokenizer
bert_inputs = bert_tokenizer(captions_list, return_tensors='pt', padding=True, truncation=True, max_length=128)

bert_model = BertModel.from_pretrained('bert-base-uncased')

def extract_bert_features(bert_inputs):
    
    # Forward pass through BERT model
    with torch.no_grad():
        outputs = bert_model(**bert_inputs)

    # Extract the output embeddings from BERT
    embeddings = outputs.last_hidden_state

    # You can use the pooled output or any other representation based on your task
    pooled_output = outputs.pooler_output

    return pooled_output


# Extract BERT features for captions
bert_features = extract_bert_features(bert_inputs)

# Now 'bert_features' contains the BERT embeddings for the captions
print("Shape of BERT features:", bert_features.shape)

 # make vocab
    vocab = deserialize_vocab(options['dataset']['vocab_path'])
    vocab_word = sorted(vocab.word2idx.items(), key=lambda x: x[1], reverse=False)
    vocab_word = [tup[0] for tup in vocab_word]

    # Create dataset, model, criterion and optimizer
    train_loader, val_loader = data.get_loaders(vocab, options)
    
    model = models.factory(options['model'],
                           vocab_word,
                           cuda=True,
                           data_parallel=False)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr=0.0002)
    

    print('Model has {} parameters'.format(utils.params_count(model)))
    # Train the Model
    best_rsum = 0
    best_score = ""
    
    print('Training has begin')

    for epoch in range(start_epoch, options['optim']['epochs']):

        utils.adjust_learning_rate(options, optimizer, epoch)

        # train for one epoch
        print("Training for the {}th epoch".format(epoch))
        engine.train(train_loader, model, optimizer, epoch, opt=options)

        #Saving the model parameters to H5 file 
        #model.save("RSITMD_model.h5") Can't do this for Pytorch models. 
        
        torch.save(model.state_dict(), "RSITMD_model.pth")
        print("Model is saved!") 
        
        # evaluate on validation set
        print("Now evaluating on the validation set")
        rsum, all_scores = engine.validate(val_loader, model)

        is_best = rsum > best_rsum
        if is_best:
            best_score = all_scores
        best_rsum = max(rsum, best_rsum)

        # save ckpt
        utils.save_checkpoint(
                {
                'epoch': epoch + 1,
                'arch': 'baseline',
                'model': model.state_dict(),
                'best_rsum': best_rsum,
                'options': options,
                'Eiters': model.Eiters,
            },
                is_best,
                filename='ckpt_{}_{}_{:.2f}.pth.tar'.format(options['model']['name'] ,epoch, best_rsum),
                prefix=options['logs']['ckpt_save_path'],
                model_name=options['model']['name']
            )

        print("Current {}th fold.".format(options['k_fold']['current_num']))
        print("Now  score:")
        print(all_scores)
        print("Best score:")
        print(best_score)