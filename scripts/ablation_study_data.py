import os
import multiprocessing
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import sys
sys.path.append("..")
from utils.vmf_batch import vMF
from models import SeqEncoder, SeqDecoder, Seq2Seq_VAE, PoolingClassifier, init_weights
from utils.training_utils import  train, evaluate, scale

    
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
N_val = 250
SEED = 17

def get_no_training_samples(k=3):
    with open('../models/ablation_study/training_data/amountOfData.txt', 'rb') as f:
        training_data = np.load(f)
    with open('../models/ablation_study/training_data/amountOfData.txt', 'wb') as f:
        np.save(f, training_data[k:])
    k = min(k, len(training_data))
    return training_data[:k]

def get_iterators(n_training_samples):
    
    folder = '3_populations'
    with open('../data/toy_data/%s/iterator/val_iterator.pkl'%folder, 'rb') as f:
        val_iterator = pickle.load(f)

    with open('../data/toy_data/%s/iterator/train_iterator_n%i.pkl'%(folder, n_training_samples), 'rb') as f:
        train_iterator = pickle.load(f)
    return train_iterator, val_iterator
        
def train_models(N_train):

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
    
    n_walks = 256
    class_fraction= 1.
    print('Training models with %i training samples...'%int(N_train))
    
    train_iterator, val_iterator = get_iterators(N_train)
        
    for k in range(1,4):
        print('Training model %i ...'%k)
        
        save_path_model =  '../models/ablation_study/training_data/vae_frac_%.2f_n%i_run%i.pt'%(class_fraction,N_train, k)
        save_path_losses = '../models/ablation_study/training_data/losses_frac_%.2f_n%i_run%i.npy'%(class_fraction,N_train,k)
        
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
            
            losses = np.load(save_path_losses)
            best_test_loss = losses[state_dict['epoch'],2]
            training = list(losses[:,:2])
            validation = list(losses[:,2:])
            last_epoch = losses.shape[0]
            
        else:
            # initialize model
            model.apply(init_weights)
            classifier.apply(init_weights)


            #optimizer
            optimizer = optim.Adam(list(model.parameters()) + list(classifier.parameters()), lr=lr)

            best_test_loss = np.infty
            training = []
            validation = []
            last_epoch = 0

        for e in range(last_epoch, N_EPOCHS):

            train_loss, train_class_loss = train(model, classifier, train_iterator, optimizer, 
                                                   calculate_loss, cross_entropy_loss, 
                                                     clip=1,norm_p=None, class_fraction=class_fraction)
            # is evaluated with 100% labels
            val_loss, val_class_loss = evaluate(model,classifier, val_iterator, 
                                                    calculate_loss, cross_entropy_loss, norm_p=None)


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
    no_training_samples = get_no_training_samples(k=NUM_PROCESSES)
    while len(no_training_samples) > 0:
        with multiprocessing.Pool(NUM_PROCESSES, maxtasksperchild=5) as pool:
            pool.map(train_models, no_training_samples)
        no_training_samples = get_no_training_samples(k=NUM_PROCESSES)