from omegaconf import OmegaConf
from hydra.experimental import compose, initialize

import nibabel as nib
import torch
import torchio as tio
import numpy as np
import random


def load_nii(img_path):

    '''
    Shortcut to load a nifti file
    '''

    nimg = nib.load(img_path)
    return nimg.get_fdata(), nimg.affine, nimg.header


def prepare_data(batch, device) : 

    inputs = batch[list(batch.keys())[0]][tio.DATA].float()
    inputs = torch.squeeze(inputs, dim = 1).to(device)

    labels = batch[list(batch.keys())[1]][tio.DATA].float()
    labels = torch.squeeze(labels, dim = 1).to(device)

    return inputs, labels


def crop_slice_zone_of_interest(volume_mask, volume_mri = None, margin = 10) : 

    '''
    Return a cropped slice 
    '''
    volume_mask = torch.tensor(volume_mask)

    non_zero_values = torch.nonzero(volume_mask, as_tuple = False)

    x_min, x_max = non_zero_values[:, 0].min(), non_zero_values[:, 0].max()
    y_min, y_max = non_zero_values[:, 1].min(), non_zero_values[:, 1].max()

    x_min, x_max = max(0, x_min - margin), min(x_max + margin, volume_mask.shape[0])
    y_min, y_max = max(0, y_min - margin), min(y_max + margin, volume_mask.shape[1])

    cropped_label = volume_mask[x_min:x_max, y_min:y_max, :]

    if volume_mri is not None : 

        volume_mri = torch.tensor(volume_mri)
        cropped_mri = volume_mri[x_min:x_max, y_min:y_max, :]
        
        return cropped_label, cropped_mri

    return cropped_label



def normalize_extreme_values(tensor, quantile_low_value = 0.01, quantile_high_value = 0.99) :

    #Make a copy to not modify the original
    tensor_to_normalize = tensor.detach().clone().float()
    quantiles = torch.tensor([quantile_low_value, quantile_high_value])

    #Compute the wanted quantiles of the tensor
    threshold_low, threshold_high = torch.quantile(tensor_to_normalize, quantiles)

    #Normalize extreme values
    tensor_to_normalize[tensor_to_normalize > threshold_high] = threshold_high
    tensor_to_normalize[tensor_to_normalize < threshold_low] = threshold_low

    return tensor_to_normalize


def sample_features_tensors(features_1, features_2, num_samples, 
                            margin = 10) :

    # Retrieve features tensors shape
    N, C = features_1.shape

    # Create sampled indices tensor
    indices = torch.randperm(features_1.shape[0])[:int(num_samples)]

    # Sampling
    sampled_tensor_1 = features_1[indices, :]
    sampled_tensor_2 = features_2[indices, :]

    return sampled_tensor_1, sampled_tensor_2
    

def generate_affine_spatial_transform(batch_size ,is_rotated = True, 
                                                  is_cropped = True, 
                                                  is_flipped = True,
                                                  is_translation = False,
                                                  max_angle = np.pi/2,
                                                  max_crop = 0.5) : 
                                                  
    identity_affine = torch.eye(3).float()
    identity_affine = identity_affine.repeat((batch_size, 1, 1))

    zeros = torch.zeros(batch_size)
    ones = torch.ones(batch_size)

    # Rotation
    if is_rotated :

        theta = torch.FloatTensor(batch_size).uniform_(-max_angle, max_angle) 
        # angles = torch.tensor([np.pi / 2, np.pi, np.pi * 3/2])
        # theta = angles[torch.multinomial(angles, batch_size, replacement = True)]

        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)

        rotation = torch.stack((torch.stack([cos_theta, -sin_theta, zeros], dim = -1), 
                               torch.stack([sin_theta, cos_theta, zeros], dim=-1),
                               torch.stack([zeros, zeros, ones], dim = -1)), dim = 1).float()

        # Remove randmly a certain percentage of the transformation
        # random_indexes = torch.randint(batch_size, (int(batch_size * 0.2), ))
        # rotation[random_indexes] = torch.eye(3)

    else : 
        rotation = identity_affine.detach().clone()


    # Cropping
    if is_cropped :

        # Define random crop parameters
        scale_factor = torch.FloatTensor(batch_size).uniform_(max_crop, 0.95) 
        translate_height = 2 * (1 - scale_factor) * torch.rand(batch_size) - (1 - scale_factor)
        translate_width = 2 * (1 - scale_factor) * torch.rand(batch_size) - (1 - scale_factor)

        # Compute copping matrix
        crop = torch.stack((torch.stack([scale_factor, zeros, translate_width], dim = -1), 
                            torch.stack([zeros, scale_factor, translate_height], dim=-1),
                            torch.stack([zeros, zeros, ones], dim = -1)), dim = 1).float()

        # Remove randmly a certain percentage of the transformation
        # random_indexes = torch.randint(batch_size, (int(batch_size * 0.2), ))
        # crop[random_indexes] = torch.eye(3)

    else : 
        crop = identity_affine.detach().clone()


    # Flip
    if is_flipped :

        horizontal_flip = torch.tensor([[1, 0, 0],
                                        [0, -1, 0],
                                        [0, 0, 1]]).float()

        flip = torch.tensor([[-1, 0, 0],
                             [0, 1, 0],
                             [0, 0, 1]]).float()

        flip = flip.repeat((batch_size, 1, 1))
        

        random_indexes = torch.randint(batch_size, (int(batch_size * 0.5), ))
        flip[random_indexes] = horizontal_flip


        # Remove randmly a certain percentage of the transformation
        random_indexes = torch.randint(batch_size, (int(batch_size * 0.3), ))
        flip[random_indexes] = torch.eye(3)

    else : 
        flip = identity_affine.detach().clone()


    # Translation
    if is_translation :

        dx = torch.FloatTensor(batch_size).uniform_(-1, 1)
        dy = torch.FloatTensor(batch_size).uniform_(-1, 1)

        translation = torch.stack((torch.stack([ones, zeros, dx], dim = -1), 
                                   torch.stack([zeros, ones, dy], dim=-1),
                                   torch.stack([zeros, zeros, ones], dim = -1)), dim = 1).float()
        
        # Remove randmly a certain percentage of the transformation
        random_indexes = torch.randint(batch_size, (int(batch_size * 0.5), ))
        translation[random_indexes] = torch.eye(3)

    else : 
        translation = identity_affine.detach().clone()



    final_affine_matrix = torch.bmm(rotation, torch.bmm(crop, torch.bmm(translation, flip)))
    final_affine_matrix = final_affine_matrix[:, :-1, :]

    return final_affine_matrix


def generate_single_affine_spatial_transform(is_rotated = True, is_cropped = True, is_flipped = True) : 

    identity_affine = torch.eye(3).float()

    if is_rotated : 

        theta = random.uniform(-np.pi/2, np.pi/2)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        rotation = torch.tensor([[cos_theta, -sin_theta, 0],
                                 [sin_theta, cos_theta, 0],
                                 [0, 0, 1]]).float()
    
    else : 
        rotation = identity_affine.clone().detach()

    
    if is_cropped :

        scale_factor = random.uniform(0.6, 0.95)
        translate_height = random.uniform(-(1 - scale_factor), (1 - scale_factor))
        translate_width = random.uniform(-(1 - scale_factor), (1 - scale_factor))

        crop = torch.tensor([[scale_factor, 0, translate_width],
                             [0, scale_factor, translate_height],
                             [0, 0, 1]]).float()

    else : 
        crop = identity_affine.clone().detach()

    
    # Flip
    if is_flipped :

        horizontal_flip = torch.tensor([[1, 0, 0],
                                        [0, -1, 0],
                                        [0, 0, 1]]).float()

        flip = torch.tensor([[-1, 0, 0],
                             [0, 1, 0],
                             [0, 0, 1]]).float()

    else : 
        flip = identity_affine.detach().clone()

        
    final_affine_matrix = torch.matmul(rotation, torch.matmul(crop, flip))
    final_affine_matrix = final_affine_matrix[:-1, :]


    return final_affine_matrix


class Config:
    def __init__(self, config_path: str = "../config"):
        self.cfg = self.load_config(config_path)

    def load_config(self, config_path: str):
        with initialize(config_path=config_path):
            cfg = compose(config_name="config")
        return cfg
