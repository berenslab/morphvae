import torch
import numpy as np

def decode_z(model,z, init_state=torch.Tensor([[0,0,0]]), max_trg_len=16):
    
    with torch.no_grad():

        # z = [batch size, LATENT DIM]
        batch_size, lat_dim = z.shape
        
        #first input to the decoder is the first coordinate
        input = init_state.repeat(batch_size,1).to(model.device)
        
        #tensor to store decoder outputs
        outputs = torch.zeros(max_trg_len, batch_size, model.decoder.output_dim).to(model.device)
        
        hidden, cell = model._get_decoder_states(z, batch_size=batch_size)
        for t in range(1, max_trg_len):
            
            #insert input token embedding, previous hidden and previous cell states
            #receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = model.decoder(input, hidden, cell)
        
            #place predictions in a tensor holding predictions for each token
            outputs[t] = output
            
            #get the highest predicted token from our predictions
            input = output
                
            # stop the decoding when only the padding element is predicted
            if input.allclose(init_state.to(model.device), atol=0.0001):
                break
    
    return outputs



def sample_rws(model, vmf, mu, max_trg_len=16, orig_seq_len=None, n_samples=4, min_angle=np.pi/3, init_state=torch.Tensor([[0,0,0]])):
    
    batch_size, latent_dim = mu.shape
    decoded_rws = []
    for k in range(n_samples):
        tup, kld, sampled_vecs = vmf.build_bow_rep(mu, n_sample=5)

        z = sampled_vecs.mean(dim=0)

        decoded_z = decode_z(model, z, max_trg_len=max_trg_len, init_state=init_state).cpu()
        seq_lengths = _get_cut_off_index_by_path_angle(decoded_z, min_angle=min_angle)
        if orig_seq_len is not None:
            seq_lengths = np.min(np.vstack((orig_seq_len, seq_lengths)), axis=0)
    
        
        decoded_z = _fill_with_infty(decoded_z, seq_lengths)
        decoded_z = decoded_z.permute(1,0,2).unsqueeze(0)
        decoded_rws.append(decoded_z)
        
    return torch.cat(decoded_rws, dim=0)


def _fill_with_infty(output, seq_len):
    """
        padds the tensor output with -infinity after the sequence length given in seq_len
        output: [walk_length, N, output_dim]
    """
    output_ = torch.ones_like(output)*np.infty*-1
    walk_length, N, output_dim = output_.shape
    for ix_ in range(N):
        l = int(seq_len[ix_])
        output_[:l, ix_, :] = output[:l, ix_,:]
    return output_

def _get_cut_off_index_by_threshold(output, thresh=2):
    
    x_co = (((output[1:,:,0] - output[:-1,:,0]).abs() >= thresh).to(int).argmax(dim=0)).unsqueeze(1)
    y_co = (((output[1:,:,1] - output[:-1,:,1]).abs() >= thresh).to(int).argmax(dim=0)).unsqueeze(1)
    z_co = (((output[1:,:,2] - output[:-1,:,2]).abs() >= thresh).to(int).argmax(dim=0)).unsqueeze(1)
    indices = torch.cat((x_co, y_co, z_co), dim=1).min(dim=1).values
    
    return indices

def _get_path_angles(trajectory):
    
    # trajectory = [n walks, walk_length, input dim]
    n_walks, walk_length, f = trajectory.shape
    
    directions = trajectory[:,1:] - trajectory[:,:-1]
    directions /= directions.norm(p=2, dim=2).reshape(n_walks,walk_length-1,1)
    
    # flatten the array so I can perform a dot product over all walks
    flattened_dirs = directions.reshape(-1,f)

    ix1 = np.array(list(set(range(1,n_walks*(walk_length-1))) - set(range(0,n_walks*walk_length,walk_length-1))))
    ix2 = np.array(list(set(range(0,n_walks*(walk_length-1))) - set(range(walk_length-2,n_walks*walk_length,walk_length-1))))

    # get the angles between the paths 
    path_angles = torch.acos((flattened_dirs[ix1]@flattened_dirs[ix2].T).diag().reshape(n_walks, -1))
    return path_angles
    
def _get_cut_off_index_by_path_angle(output, min_angle=np.pi/2):
    
    # output = [walk_length, n_walks, input dim]
    walk_length, n_walks, f = output.shape
    output = output.transpose(0,1)
   
    path_angles = _get_path_angles(output)
    non_zeros = (path_angles[:,1:] >= min_angle).nonzero()
       
    indices = torch.ones((n_walks), dtype=torch.long)*walk_length
    for x,y in non_zeros:
        indices[x] = min(indices[x], y+2) # shift by 1 because we used the diff in the beginning
    
    
    return indices

