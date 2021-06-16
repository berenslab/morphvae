import torch
import numpy as np
import sys
sys.path.append("..")
from models import Seq2SeqDataSet, SeqEncoder, SeqDecoder, Seq2Seq_VAE
from .vmf_batch import vMF

SEED = 17

def scale(X,scale=(.1,3)):
    a = scale[0]
    b = scale[1]
    s = a + (b-a)*torch.rand(1)
    return X*s

def create_Seq2SeqDataset(walks, labels=None, MASKING_ELEMENT = 0, transform=None):
    N, n_walks, walk_length, _ = walks.shape
    RW_list = []
    for n in range(N):

        rws = []
        for w in range(n_walks):
            single_walk = walks[n,w]
            single_walk = single_walk[single_walk != np.infty*-1]
            rws.append(torch.Tensor(single_walk).view(-1,3))
        RW_list.append(rws)
    
    SeqDS = Seq2SeqDataSet(RW_list, labels, max_length=walk_length, n_walks=n_walks, output_dim=3, 
                           masking_element=MASKING_ELEMENT, transform=transform)
    return SeqDS


def create_model(config, device):
    INPUT_DIM = config['input_dim']
    EMBED_DIM = config['embed_dim']
    HIDDEN_DIM = config['hidden_dim']
    LATENT_DIM = config['latent_dim']
    NUM_LAYERS = config['num_layers']
    KAPPA = config['kappa']
    DROPOUT = config['dropout']
    

    # model
    enc = SeqEncoder(INPUT_DIM, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
    dec = SeqDecoder(INPUT_DIM, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
    dist = vMF(LATENT_DIM, kappa=KAPPA)
    model = Seq2Seq_VAE(enc, dec, dist, device).to(device)
    return model

def train(model,classifier, iterator, optimizer, model_criterion, 
                          classifier_criterion, clip, norm_p=1, class_fraction=1., ignore_index=-100):
    
    model.train()
    device = model.device
    epoch_loss = 0
    np.random.seed(SEED)
    for i, (src_batch, trg_batch, seq_len, ix, true_labels) in enumerate(iterator):
        
        # src_batch = [batch size, n_walks, walk length, input dim]
        bs, n_walks, walk_length, input_dim = src_batch.shape
        
        src = src_batch.view(-1,walk_length,input_dim).transpose(0,1).to(device)
        trg = trg_batch.view(-1, walk_length, input_dim).transpose(0,1).to(device)
        src_len = seq_len.view(-1).to(device)
        true_labels = true_labels.to(device)
        
#         GPUtil.showUtilization()
        # src = [walk length, batch size * n_walks, input dim]

        optimizer.zero_grad()
        
        output = model(src, src_len, trg)
        
        #trg = [trg len, batch size * n walks, output dim]
        #output = [trg len, batch size  * n walks, output dim]
        
        rec_loss = model_criterion(output, trg)
        
        ## add the classification part           
    
        k = int(bs*class_fraction)
        unlabeled_examples = np.random.choice(range(bs), size=bs-k, replace=False)
        # unlabeled_examples contains the indices of the unlabeled fraction of data

        # set the unlabelled examples to ignore index
        masked_labels = torch.zeros_like(true_labels)
        masked_labels.copy_(true_labels)
        masked_labels[unlabeled_examples] = ignore_index
       
        pred_label = classifier(model.Z)

        class_vae_loss = classifier_criterion(pred_label,masked_labels) * n_walks 

        # if the no or little labels are passed to the classifier it should still be able to train
        Z_ = model.Z.detach()
        pred_label_ = classifier(Z_)

        class_loss = classifier_criterion(pred_label_,true_labels) * n_walks 
        class_loss.backward()
        
        if norm_p is not None:
            norm_loss = model.h.norm(p=norm_p, dim=1).sum()
        else: 
            norm_loss = 0
        loss = rec_loss + class_vae_loss + norm_loss
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        torch.nn.utils.clip_grad_norm_(classifier.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
    
    torch.cuda.empty_cache()
    return epoch_loss / len(iterator), class_loss.item()/len(iterator)


def evaluate(model, classifier, iterator, model_criterion, classifier_criterion, norm_p=1):
    
    model.eval()
    device = model.device
    epoch_loss = 0
    
    with torch.no_grad():
    
        for i, (src_batch, trg_batch, seq_len, ix, true_labels) in enumerate(iterator):
            
             # src_batch = [batch size, n_walks, walk length, input dim]
            bs, n_walks, walk_length, input_dim = src_batch.shape
            src = src_batch.view(-1,walk_length,input_dim).transpose(0,1).to(device)
            trg = trg_batch.view(-1, walk_length, input_dim).transpose(0,1).to(device)
            src_len = seq_len.view(-1).to(device)
            true_labels = true_labels.to(device)
            
            output = model(src,src_len, trg, 0) #turn off teacher forcing

            #trg = [trg len, batch size* n walks, outpur dim]
            #output = [trg len, batch size * n walks, output dim]

            rec_loss = model_criterion(output, trg)
            
             # add the classification part.
            pred_label = classifier(model.Z)

            class_loss = classifier_criterion(pred_label,true_labels) * n_walks    
            
            if norm_p is not None:
                norm_loss = model.h.norm(p=norm_p, dim=1).sum()
            else: 
                norm_loss = 0
            
            loss = rec_loss + class_loss + norm_loss
        
            epoch_loss += loss.item()
    torch.cuda.empty_cache()
    return epoch_loss / len(iterator), class_loss.item()/len(iterator)
