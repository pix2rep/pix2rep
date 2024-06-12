import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchio as tio
import monai

import clean_code.pix2rep.utils as utils
import clean_code.pix2rep.losses as losses
from clean_code.pix2rep.models import U_Net_CL, MLP


class CL_Model:

    def __init__(self, cfg) :

        self.cfg = cfg
        self.device = torch.device("cuda"  if torch.cuda.is_available() else "cpu")

        self.loss_function = nn.CrossEntropyLoss().to(self.device) 
        self.evaluation_loss = monai.losses.DiceLoss(include_background = True, 
                                            to_onehot_y = False,
                                            reduction = 'mean',
                                            softmax = True).to(self.device)

        self.model = U_Net_CL.UNet(self.cfg.contrastive_pretraining.n_channels, self.cfg.contrastive_pretraining.n_features_map).to(self.device) 
        self.finetuning_layer =  U_Net_CL.OutConv(self.cfg.contrastive_pretraining.n_features_map, self.cfg.contrastive_pretraining.n_classes).to(self.device)

        if self.cfg.contrastive_pretraining.projection_head_depth == 0 :
             self.projection_head = MLP.Identity()
             
        elif self.cfg.contrastive_pretraining.projection_head_depth == 1 : 
            self.projection_head = MLP.MLP(channels_in = self.cfg.contrastive_pretraining.n_features_map, 
                            channels_out = self.cfg.contrastive_pretraining.n_features_map_mlp, 
                            inner_dim_1 = self.cfg.contrastive_pretraining.inner_dim_1,
                            ).to(self.device)
        
        elif self.cfg.contrastive_pretraining.projection_head_depth == 2 : 
            self.projection_head = MLP.ConvMLP_3_layers(channels_in = self.cfg.contrastive_pretraining.n_features_map, 
                              channels_out = self.cfg.contrastive_pretraining.n_features_map_mlp, 
                              inner_dim_1 = self.cfg.contrastive_pretraining.inner_dim_1,
                              inner_dim_2 = self.cfg.contrastive_pretraining.inner_dim_2,
                              ).to(self.device)

        elif self.cfg.contrastive_pretraining.projection_head_depth == 3 :
            self.projection_head = MLP.ConvMLP_4_layers(
                              channels_in = self.cfg.contrastive_pretraining.n_features_map, 
                              channels_out = self.cfg.contrastive_pretraining.n_features_map_mlp, 
                              inner_dim_1 = self.cfg.contrastive_pretraining.inner_dim_1,
                              inner_dim_2 = self.cfg.contrastive_pretraining.inner_dim_2,
                              inner_dim_3 = self.cfg.contrastive_pretraining.inner_dim_3
                              ).to(self.device)


        if self.cfg.contrastive_pretraining.weights_backbone_load_path != None : 
            self.model.load_state_dict(torch.load(self.cfg.contrastive_pretraining.weights_backbone_load_path))
            print('backbone loaded')

        if self.cfg.contrastive_pretraining.weights_ft_layer_load_path != None : 
            self.finetuning_layer.load_state_dict(torch.load(self.cfg.contrastive_pretraining.weights_ft_layer_load_path))
            print('outconv loaded')


    def load_backbone_model(self, weights_load_path) :
        if weights_load_path != None:
            self.model.load_state_dict(torch.load(weights_load_path))


    def load_outconv_model(self, weights_load_path) :
        if weights_load_path != None:
            self.finetuning_layer.load_state_dict(torch.load(weights_load_path))


    def save_best_model(self, validation_losses, model, save_path) :

        if validation_losses[-1] <= np.min(validation_losses[:-1]) :
            torch.save(model.state_dict(), save_path)
            # print('model saved')

    
    def early_stopping(self, validation_losses) : 
        if len(validation_losses) - validation_losses.index(np.min(validation_losses)) > 6 :
            return True
        else :
            return False

    def run_training(self, training_loader_CL, validation_loader_CL) : 

        avg_train_losses = []
        avg_val_losses = []

        optimizer = torch.optim.Adam(list(self.model.parameters()) + list(self.projection_head.parameters()), lr=self.cfg.contrastive_pretraining.learning_rate_backbone)

        for epoch in range(self.cfg.contrastive_pretraining.num_epochs) :

            train_loss = []
            val_loss = []

            # Training
            self.model.train()
            self.projection_head.train()
            with tqdm.tqdm(training_loader_CL, unit = 'batch', disable = self.cfg.contrastive_pretraining.tqdm_disabled) as tepoch :
                for batch_index, batch in enumerate(tepoch) :

                    # Prepare Data
                    view_1 = batch[0].squeeze(1).float().to(self.device)
                    view_2 = batch[1].squeeze(1).float().to(self.device)

                    transfo_affine = utils.generate_affine_spatial_transform(batch[0].shape[0] ,
                                                        self.cfg.contrastive_pretraining.is_rotated, 
                                                        self.cfg.contrastive_pretraining.is_cropped, 
                                                        self.cfg.contrastive_pretraining.is_flipped,
                                                        self.cfg.contrastive_pretraining.is_translation,
                                                        self.cfg.contrastive_pretraining.max_angle,
                                                        self.cfg.contrastive_pretraining.max_crop).to(self.device)
                    # Affine Spatial Transformation
                    grid = F.affine_grid(transfo_affine, view_2.shape, align_corners = True)
                    view_2 = F.grid_sample(view_2, grid, align_corners = True)

                    # Forward pass of the 2 views through the network and projection head
                    features_1 = self.projection_head(self.model(view_1))  
                    features_2 = self.projection_head(self.model(view_2))

                    # Apply spatial transformation to the feature map
                    grid = F.affine_grid(transfo_affine, features_1.shape, align_corners = True)
                    features_1 = F.grid_sample(features_1, grid, align_corners = True)
                
                    # Compute logits and loss
                    logits, labels = losses.info_nce_loss(features_1, features_2, 
                                                          self.cfg.contrastive_pretraining.num_samples, 
                                                          self.cfg.contrastive_pretraining.temperature, 
                                                          self.device)
                    batch_loss_training = self.loss_function(logits, labels)
                    train_loss.append(batch_loss_training.item())
                    
                    # Reset Gradients
                    optimizer.zero_grad()
                    batch_loss_training.backward(retain_graph = True)
                    optimizer.step()

                    #Logging
                    tepoch.set_description(f"Epoch {epoch}")
                    tepoch.set_postfix(training_loss = f'{batch_loss_training.item()}')     

            avg_train_losses.append(np.average(train_loss))

            # Validation
            if epoch % self.cfg.contrastive_pretraining.eval_frequency == 0 :
                self.model.eval()
                self.projection_head.eval()
                with torch.no_grad() :
                    with tqdm.tqdm(validation_loader_CL, unit = 'batch', disable = self.cfg.contrastive_pretraining.tqdm_disabled) as tepoch :
                        for batch_index_val, batch_val in enumerate(tepoch) :

                            # Prepare Data
                            view_1 = batch_val[0].squeeze(1).float().to(self.device)
                            view_2 = batch_val[1].squeeze(1).float().to(self.device)

                            transfo_affine = utils.generate_affine_spatial_transform(batch_val[0].shape[0] ,
                                                        self.cfg.contrastive_pretraining.is_rotated, 
                                                        self.cfg.contrastive_pretraining.is_cropped, 
                                                        self.cfg.contrastive_pretraining.is_flipped,
                                                        self.cfg.contrastive_pretraining.is_translation,
                                                        self.cfg.contrastive_pretraining.max_angle,
                                                        self.cfg.contrastive_pretraining.max_crop).to(self.device)

                            # Affine Spatial Transformation
                            grid = F.affine_grid(transfo_affine, view_2.shape, align_corners = True)
                            view_2 = F.grid_sample(view_2, grid, align_corners = True)

                            # Forward pass of the 2 views through the network and projection head
                            features_1 = self.projection_head(self.model(view_1)) 
                            features_2 = self.projection_head(self.model(view_2))

                            # Apply spatial transformation to the feature map
                            features_1 = F.grid_sample(features_1, grid, align_corners = True)

                            # Compute logits and loss
                            logits, labels = losses.info_nce_loss(features_1, features_2, self.cfg.contrastive_pretraining.num_samples, self.cfg.contrastive_pretraining.temperature, self.device)
                            batch_loss_validation = self.loss_function(logits, labels)

                            val_loss.append(batch_loss_validation.item())

                            # Logging
                            tepoch.set_description(f"Epoch {epoch}")
                            tepoch.set_postfix(validation_loss = f'{batch_loss_validation.item()}')       
                            
                        avg_val_losses.append(np.average(val_loss))
                        
                        if len(avg_val_losses) == 1 :
                            torch.save(self.model.state_dict(), self.cfg.contrastive_pretraining.save_path_backbone)
                        else :
                            self.save_best_model(avg_val_losses, self.model, self.cfg.contrastive_pretraining.save_path_backbone)


        return avg_train_losses, avg_val_losses


    def run_linear_probing(self, training_loader, validation_loader) :

        avg_train_losses = []
        avg_val_losses = []
        
        optimizer = torch.optim.Adam(self.finetuning_layer.parameters(), lr=self.cfg.contrastive_pretraining.linear_probing_learning_rate_outconv)

        for epoch in range(self.cfg.contrastive_pretraining.num_epochs) :

            train_loss = []
            val_loss = []

            # Training
            self.model.eval()
            self.finetuning_layer.train()
            with tqdm.tqdm(training_loader, unit = 'batch', disable = self.cfg.contrastive_pretraining.tqdm_disabled) as tepoch :
                for batch_index, batch in enumerate(tepoch) :

                    inputs = batch[0].squeeze(1).float().to(self.device)
                    labels = batch[1].squeeze(1).float().to(self.device)

                    labels = losses.transform_mask_for_dice_loss(labels, batch).to(self.device)

                    logits = self.finetuning_layer(self.model(inputs))
                    
                    batch_loss_training = self.evaluation_loss(logits, labels)
                    train_loss.append(batch_loss_training.item())
                    
                    optimizer.zero_grad()
                    batch_loss_training.backward(retain_graph = True)
                    optimizer.step()

                    #Logging
                    tepoch.set_description(f"Epoch {epoch}")
                    tepoch.set_postfix(training_loss = f'{batch_loss_training.item()}')      

            avg_train_losses.append(np.average(train_loss))
            

            # Validation
            if epoch % self.cfg.contrastive_pretraining.eval_frequency == 0 :
                self.model.eval()
                self.finetuning_layer.eval()
                with torch.no_grad() :
                    with tqdm.tqdm(validation_loader, unit = 'batch', disable = self.cfg.contrastive_pretraining.tqdm_disabled) as tepoch :
                        for batch_index_val, batch_val in enumerate(tepoch) :

                            inputs = batch_val[0].squeeze(1).float().to(self.device)
                            labels = batch_val[1].squeeze(1).float().to(self.device)
                            
                            labels = losses.transform_mask_for_dice_loss(labels, batch_val).to(self.device)

                            logits = self.finetuning_layer(self.model(inputs))
                            
                            batch_loss_validation = self.evaluation_loss(logits, labels)
                            val_loss.append(batch_loss_validation.item())

                            #Logging
                            tepoch.set_description(f"Epoch {epoch}")
                            tepoch.set_postfix(validation_loss = f'{batch_loss_validation.item()}')
                            
                        avg_val_losses.append(np.average(val_loss))

                        if len(avg_val_losses) == 1 :
                            torch.save(self.finetuning_layer.state_dict(), self.cfg.contrastive_pretraining.save_path_outconv_layer)
                        else :
                            self.save_best_model(avg_val_losses, self.finetuning_layer, self.cfg.contrastive_pretraining.save_path_outconv_layer)


        return avg_train_losses, avg_val_losses

    
    def run_finetuning(self, training_loader, validation_loader) :

        avg_train_losses = []
        avg_val_losses = []

        optim_backbone = torch.optim.Adam(self.model.parameters(), lr=self.cfg.contrastive_pretraining.finetuning_learning_rate_backbone)
        optim_outconv = torch.optim.Adam(self.finetuning_layer.parameters(), lr=self.cfg.contrastive_pretraining.finetuning_learning_rate_outconv)

        for epoch in range(self.cfg.contrastive_pretraining.num_epochs) :

            train_loss = []
            val_loss = []

            # Training
            self.model.train()
            self.finetuning_layer.train()
            with tqdm.tqdm(training_loader, unit = 'batch', disable = self.cfg.contrastive_pretraining.tqdm_disabled) as tepoch :
                for batch_index, batch in enumerate(tepoch) :

                    inputs = batch[0].squeeze(1).float().to(self.device)
                    labels = batch[1].squeeze(1).float().to(self.device)

                    labels = losses.transform_mask_for_dice_loss(labels, batch).to(self.device)

                    logits = self.finetuning_layer(self.model(inputs))
                    
                    batch_loss_training = self.evaluation_loss(logits, labels)
                    train_loss.append(batch_loss_training.item())
                    
                    optim_backbone.zero_grad()
                    optim_outconv.zero_grad()

                    batch_loss_training.backward(retain_graph = True)

                    optim_backbone.step()
                    optim_outconv.step()

                    #Logging
                    tepoch.set_description(f"Epoch {epoch}")
                    tepoch.set_postfix(training_loss = f'{batch_loss_training.item()}')      

            avg_train_losses.append(np.average(train_loss))
            

            # Validation
            if epoch % self.cfg.contrastive_pretraining.eval_frequency == 0 :
                self.model.eval()
                self.finetuning_layer.eval()
                with torch.no_grad() :
                    with tqdm.tqdm(validation_loader, unit = 'batch', disable = self.cfg.contrastive_pretraining.tqdm_disabled) as tepoch :
                        for batch_index_val, batch_val in enumerate(tepoch) :

                            inputs = batch_val[0].squeeze(1).float().to(self.device)
                            labels = batch_val[1].squeeze(1).float().to(self.device)
                            
                            labels = losses.transform_mask_for_dice_loss(labels, batch_val).to(self.device)

                            logits = self.finetuning_layer(self.model(inputs))
                            
                            batch_loss_validation = self.evaluation_loss(logits, labels)
                            val_loss.append(batch_loss_validation.item())

                            #Logging
                            tepoch.set_description(f"Epoch {epoch}")
                            tepoch.set_postfix(validation_loss = f'{batch_loss_validation.item()}')
                            
                        avg_val_losses.append(np.average(val_loss))

                        if len(avg_val_losses) == 1 :
                            # torch.save(self.model.state_dict(), self.cfg.contrastive_pretraining.save_path_outconv_layer)
                            pass
                        else :
                            self.save_best_model(avg_val_losses, self.model, self.cfg.contrastive_pretraining.save_path_backbone)
                            self.save_best_model(avg_val_losses, self.finetuning_layer, self.cfg.contrastive_pretraining.save_path_outconv_layer)
                            
                if self.early_stopping(avg_val_losses) : 
                    print(f'Fine Tuning Training Early Stopping : Epoch n° {epoch}')
                    break

        return avg_train_losses, avg_val_losses

    
    def run_test(self, testing_loader) :

        test_losses = []

        self.model.eval()
        self.finetuning_layer.eval()
        with torch.no_grad() :
            with tqdm.tqdm(testing_loader, unit = 'batch', disable = self.cfg.contrastive_pretraining.tqdm_disabled) as tepoch :
                for batch_index, batch in enumerate(tepoch) :

                    inputs = batch[0].squeeze(1).float().to(self.device)
                    labels = batch[1].squeeze(1).float().to(self.device)
                    
                    labels = losses.transform_mask_for_dice_loss(labels, batch).to(self.device)

                    logits = self.finetuning_layer(self.model(inputs))
                    
                    batch_loss_testing = self.evaluation_loss(logits, labels)#.mean(dim = 0)

                    test_losses.append(batch_loss_testing.item())#[:, 0, 0].cpu())

        return test_losses

    def run_test_volume(self, testing_loader_volume) :

        test_losses = []

        loss_function = monai.losses.DiceLoss(include_background = True,
                                            to_onehot_y = False,
                                            reduction = 'none',
                                            softmax = True)

        self.model.eval()
        self.finetuning_layer.eval()
        with torch.no_grad() :
            with tqdm.tqdm(testing_loader_volume, unit = 'batch', disable = self.cfg.contrastive_pretraining.tqdm_disabled) as tepoch :
                for batch_index, batch in enumerate(tepoch) :

                    inputs = batch[0].squeeze(0).float().to(self.device)
                    labels = batch[1].permute(0, 2, 3, 4, 1)
                    
                    labels = losses.transform_mask_for_dice_loss_3D(labels, batch).to(self.device)

                    logits = self.finetuning_layer(self.model(inputs))
                    logits = logits.permute(1, 2, 3, 0).unsqueeze(0)
                    
                    batch_loss_testing = loss_function(logits, labels).mean(dim = 0)

                    test_losses.append(batch_loss_testing[:, 0, 0].cpu()) #.item())#)

        test_losses_detailed = torch.stack(test_losses).mean(dim = 0)
        test_losses = torch.mean(test_losses_detailed)

        return test_losses, test_losses_detailed

    def run_detailed_test(self, testing_loader) :

        test_losses = []
        loss_detailed = monai.losses.DiceLoss(include_background = True,
                                        to_onehot_y = False,
                                        reduction = 'none',
                                        softmax = True).to(self.device)

        self.model.eval()
        self.finetuning_layer.eval()
        with torch.no_grad() :
                    with tqdm.tqdm(testing_loader, unit = 'batch', disable = self.cfg.contrastive_pretraining.tqdm_disabled) as tepoch :
                        for batch_index, batch in enumerate(tepoch) :

                            inputs = batch[0].squeeze(1).float().to(self.device)
                            labels = batch[1].squeeze(1).float().to(self.device)
                            
                            labels = losses.transform_mask_for_dice_loss(labels, batch).to(self.device)

                            logits = self.finetuning_layer(self.model(inputs))
                            
                            batch_loss_testing = loss_detailed(logits, labels).mean(dim = 0)

                            test_losses.append(batch_loss_testing[:, 0, 0].cpu())

        return test_losses

                    

