import torch
import torch.nn.functional as F
import torchio as tio

import einops as ops

import utils

# def transform_mask_for_dice_loss(labels, batch, num_classes = 4, img_size = 64) :

#     batch_size = batch['mri_slice'][tio.DATA].shape[0]
#     new_masks = torch.zeros([batch_size, num_classes, img_size, img_size])
    
#     for value in labels.unique() :
#         new_masks[:, int(value):int(value+1), :, :][labels == value] = 1

#     return new_masks

def transform_mask_for_dice_loss(labels, batch, num_classes = 4) :

    batch_size = batch[0].shape[0]
    img_size = batch[0].shape[-1]

    new_masks = torch.zeros([batch_size, num_classes, img_size, img_size])
    
    for value in labels.unique() :
        new_masks[:, int(value):int(value+1), :, :][labels == value] = 1

    return new_masks

def transform_mask_for_dice_loss_3D(labels, batch) :

    batch_size = 1
    num_classes = 4
    img_size = 128
    depth = labels.shape[-1]
    
    new_masks = torch.zeros([batch_size, num_classes, img_size, img_size, depth])
    
    for value in labels.unique() :
        new_masks[:, int(value):int(value+1), :, :, :][labels == value] = 1

    return new_masks


def info_nce_loss(features_1, features_2, num_samples, temperature, device) :

    num_samples = int(min(num_samples, features_1.shape[0] * features_1.shape[2] * features_1.shape[3]))
    
    # Labels
    batch_targets = torch.cat((torch.arange(num_samples - 1, 2*num_samples - 1), 
                               torch.arange(num_samples)),
                               dim = 0).to(device)

    # Create matrices
    features_1 = ops.rearrange(features_1, 'b c h w -> (b h w) c')
    features_2 = ops.rearrange(features_2, 'b c h w -> (b h w) c')

    # Normalize the feature vectors across the channels dimension
    features_1 = F.normalize(features_1, dim=1) 
    features_2 = F.normalize(features_2, dim=1)

    # Sample features vectors
    features_1, features_2 = utils.sample_features_tensors(features_1, features_2, num_samples)
   

    logits = torch.cat((features_1, features_2), dim = 0)

    similarity_matrix = torch.matmul(logits, logits.T)

    mask = torch.eye(len(batch_targets)).to(device)
    logits = similarity_matrix[~mask.bool()].view(similarity_matrix.shape[0], -1)
    logits = logits / temperature

    return logits, batch_targets

# def info_nce_loss_simclr(features_1, features_2, temperature, device) :

#     labels = torch.cat([torch.arange(features_1.shape[0]) for i in range(2)], dim=0)
#     labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
#     labels = labels.to(device)

#     # Create matrices
#     features_1 = ops.rearrange(features_1, 'b c h w -> (b h w) c')
#     features_2 = ops.rearrange(features_2, 'b c h w -> (b h w) c')

#     # Normalize the feature vectors across the channels dimension
#     features_1 = F.normalize(features_1, dim=1) 
#     features_2 = F.normalize(features_2, dim=1)

#     logits = torch.cat((features_1, features_2), dim = 0)
#     similarity_matrix = torch.matmul(logits, logits.T)

#     mask = torch.eye(labels.shape[0]).to(device)
#     logits = similarity_matrix[~mask.bool()].view(similarity_matrix.shape[0], -1)
#     logits = logits / temperature

#     return logits, batch_targets

def info_nce_loss_simclr(features_1, features_2, temperature, device) :

    batch_targets = torch.cat((torch.arange(features_1.shape[0] - 1, 2*features_1.shape[0] - 1), 
                               torch.arange(features_1.shape[0])),
                               dim = 0).to(device)


    # Create matrices
    features_1 = ops.rearrange(features_1, 'b c h w -> (b h w) c')
    features_2 = ops.rearrange(features_2, 'b c h w -> (b h w) c')

    # Normalize the feature vectors across the channels dimension
    features_1 = F.normalize(features_1, dim=1) 
    features_2 = F.normalize(features_2, dim=1)

    logits = torch.cat((features_1, features_2), dim = 0)
    similarity_matrix = torch.matmul(logits, logits.T)

    mask = torch.eye(len(batch_targets)).to(device)
    logits = similarity_matrix[~mask.bool()].view(similarity_matrix.shape[0], -1)
    logits = logits / temperature

    return logits, batch_targets

def grouped_loss(features_1, features_2, num_groups, num_samples, temperature) :

    # Rearrange features vectors
    features_1 = ops.rearrange(features_1, 'b c h w -> (b h w) c')
    features_2 = ops.rearrange(features_2, 'b c h w -> (b h w) c')

    # Normalize the feature vectors
    features_1 = F.normalize(features_1, dim=1) #Across channels ou across images ?
    features_2 = F.normalize(features_2, dim=1)

    # Shuffle features vectors
    index_shuffled = torch.randperm(features_1.shape[0])
    features_1 = features_1[index_shuffled, :][:num_samples]
    features_2 = features_2[index_shuffled, :][:num_samples]

    # Division in subgroups
    features_1 = ops.rearrange(features_1, '(g n) c -> g n c', g = num_groups, c = features_1.shape[-1])
    features_2 = ops.rearrange(features_2, '(g n) c -> g n c', g = num_groups, c = features_2.shape[-1])

    # Compute mask to filter similarity matrix
    mask = torch.eye(2 * features_1.shape[1]).repeat(num_groups, 1, 1).bool().to(device)

    # Compute batched similarity matrix
    logits = torch.cat((features_1, features_2), dim = 1)
    similarity_matrix = torch.bmm(logits, logits.permute(0, 2, 1))[~mask]

    logits = ops.rearrange(similarity_matrix, '(g n m) -> g n m', g = num_groups, n = features_1.shape[1])
    logits = logits / temperature

    batch_targets = torch.cat((torch.arange(features_1.shape[1] - 1, 2*features_1.shape[1] - 1), 
                               torch.arange(features_1.shape[1])),
                               dim = 0)
    batch_targets = batch_targets.repeat(num_groups, 1).to(device)

    return logits, batch_targets



def dice_coefficient(output, target, epsilon=1e-5):
    # Reshape predictions and targets for batch-wise calculations
    output = output.view(output.size(0), output.size(1), -1)
    target = target.view(target.size(0), target.size(1), -1)

    intersection = torch.sum(output * target, dim=2)
    union = torch.sum(output, dim=2) + torch.sum(target, dim=2)

    dice_coeff = (2.0 * intersection + epsilon) / (union + epsilon)

    return dice_coeff


def dice_loss(output, target, use_hard_pred = False):

    if use_hard_pred:
        # This casts the predictions to binary 0 or 1
        output = F.one_hot(output.argmax(dim = 1), num_classes = 4 ).permute(0, 3, 1, 2)
    
    return 1 - dice_coefficient(output, target)