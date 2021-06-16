import os
import multiprocessing
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler

import sys
sys.path.append("..")
from utils.vmf_batch import vMF
from models import SeqEncoder, SeqDecoder, Seq2Seq_VAE, PoolingClassifier, init_weights
from utils.training_utils import create_Seq2SeqDataset, train, evaluate
from utils.rw_utils import load_neurons

SEED = 17
BATCH_SIZE = 128                  # number of data points in each batch
N_train = 750
N_val = 250 
    
# parameter
INPUT_DIM = 3   
EMB_DIM = 32
HID_DIM = 32
LATENT_DIM = 32
DROPOUT = 0.1
KAPPA = 500
lr = 1e-2                           # learning rate
NUM_LAYERS = 2
NUM_CLASSES = 3
POOL = 'max'
N_EPOCHS = 150

def get_label_fractions(k=3):
    with open('../models/classification_analysis/k500/amountOfLabels.txt', 'rb') as f:
        label_fractions = np.load(f)
    with open('../models/classification_analysis/k500/amountOfLabels.txt', 'wb') as f:
        np.save(f, label_fractions[k:])
    k = min(k, len(label_fractions))
    return label_fractions[:k]

def create_iterators():
    # load in data
    neurons = load_neurons('../data/toy_data/neurons/')

    with open('../data/toy_data/walk_representation.npy', 'rb') as f:
        walk_representation = np.load(f)
    
    # create Seq2Seq data set
    MASKING_ELEMENT = 0
    true_labels = torch.Tensor([0]*400 + [1]*400 + [2]*400).to(torch.long)
    SeqDS = create_Seq2SeqDataset(walk_representation, true_labels,MASKING_ELEMENT)

    # get training, validation, test split

    np.random.seed(SEED)
    N = walk_representation.shape[0]

    train_index = np.random.choice(range(N), size=N_train, replace=False)
    val_index = np.random.choice(list(set(range(N)) - set(train_index)), size=N_val, replace=False)

    train_sampler = SubsetRandomSampler(train_index)
    val_sampler = SubsetRandomSampler(val_index)

    # Create iterators
    train_iterator = torch.utils.data.DataLoader(SeqDS, batch_size=BATCH_SIZE, 
                                           sampler=train_sampler)
    val_iterator = torch.utils.data.DataLoader(SeqDS, batch_size=len(val_index), 
                                           sampler=val_sampler)
    return train_iterator, val_iterator

def train_models(class_fraction):

    def calculate_loss(x, reconstructed_x, ignore_el=0):
        # reconstruction loss
        # x = [trg len, batch size * n walks, output dim]

        seq_len , bs, output_dim = x.shape
        mask = x[:,:,0] != ignore_el
        RCL = 0
        for d in range(output_dim):
            RCL += mse_loss(reconstructed_x[:,:,d][mask], x[:,:,d][mask])
        RCL /= output_dim
    
        return RCL

    # train three models for each fraction of labels used
    torch.manual_seed(SEED)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_iterator, val_iterator = create_iterators()
    
    n_walks = 256
    
    print('Training models with %i percent of the labels...'%int(class_fraction*100))
        
        
    for k in range(1,4):
        print('Training model %i ...'%k)
        
        save_path_model =  '../models/classification_analysis/k500/vae_frac_%.2f_run%i.pt'%(class_fraction, k)
        save_path_losses = '../models/classification_analysis/k500/losses_frac_%.2f_run%i.npy'%(class_fraction,k)
        
        enc = SeqEncoder(INPUT_DIM, EMB_DIM, HID_DIM, NUM_LAYERS, DROPOUT)
        dec = SeqDecoder(INPUT_DIM, EMB_DIM, HID_DIM, NUM_LAYERS, DROPOUT)
        dist = vMF(LATENT_DIM, kappa=KAPPA)
        model = Seq2Seq_VAE(enc, dec, dist, device).to(device)
        classifier = PoolingClassifier(LATENT_DIM, NUM_CLASSES, n_walks,DROPOUT,pooling=POOL).to(device)

        # losses
        cross_entropy_loss = nn.CrossEntropyLoss(reduction='sum')
        mse_loss = nn.MSELoss(reduction='sum')
        
        if os.path.exists(save_path_model):
            
            # load model and train further
            state_dict = torch.load(save_path_model)
            model.load_state_dict(state_dict['model_state_dict'])
            classifier.load_state_dict(state_dict['classifier_state_dict'])
            
            optimizer = optim.Adam(list(model.parameters()) + list(classifier.parameters()))
            optimizer.load_state_dict(state_dict['optimizer_state_dict'])
            # half the learning rate to be consistent with the other training
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']/2
            
            losses = np.load(save_path_losses)
            best_test_loss = losses[state_dict['epoch'],2]
            training = list(losses[:,:2])
            validation = list(losses[:,2:])
            
        else:
            # initialize model
            model.apply(init_weights)
            classifier.apply(init_weights)


            #optimizer
            optimizer = optim.Adam(list(model.parameters()) + list(classifier.parameters()), lr=lr)

            best_test_loss = np.infty
            training = []
            validation = []

        for e in range(N_EPOCHS):

            train_loss, train_class_loss = train(model, classifier, train_iterator, optimizer, 
                                                   calculate_loss, cross_entropy_loss, 
                                                     1, class_fraction)
            # is evaluated with 100% labels
            val_loss, val_class_loss = evaluate(model,classifier, val_iterator, 
                                                    calculate_loss, cross_entropy_loss)


            train_loss /= N_train
            train_class_loss /= N_train
            val_loss /= N_val
            val_class_loss /=N_val

            training += [[train_loss, train_class_loss]]
            validation += [[val_loss, val_class_loss]]
            #print(f'Epoch {e}, Train Loss: {train_loss:.2f}, Test Loss: {val_loss:.2f}')

            if e % 50 == 0 and e > 0:
                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']/2

            if best_test_loss > val_loss:
                best_test_loss = val_loss
                torch.save({'epoch': e,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'classifier_state_dict': classifier.state_dict()
                                   },save_path_model)

                validation_ = np.array(validation)
                training_ = np.array(training)
                # [:,0] = training loss, [:,1] = training classification loss 
                # [:,2] validation loss, [:,3] validation classification loss
                losses = np.hstack((training_, validation_))
                np.save(save_path_losses,losses)
                
        validation = np.array(validation)
        training = np.array(training)
        # [:,0] = training loss, [:,1] = training classification loss 
        # [:,2] validation loss, [:,3] validation classification loss
        losses = np.hstack((training, validation))
        np.save(save_path_losses, losses)

if __name__ == "__main__":
    
    # run the classification analysis multiple times
    NUM_PROCESSES = 3
    fractions = get_label_fractions(k=NUM_PROCESSES)
    while len(fractions) > 0:
        with multiprocessing.Pool(NUM_PROCESSES) as pool:
            pool.map(train_models, fractions)
        fractions = get_label_fractions(k=NUM_PROCESSES)