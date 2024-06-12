import os
import glob
import logging

import numpy as np
import random

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchio as tio
from torchvision import transforms

import utils


class ACDC_dataset:
    def __init__(self, data_folder_path):
        self.data_folder_path = data_folder_path

    
    def retrieve_patients_infos(self) :

        """Retrieve a dictionary with patients information"""

        patients_info = {'training' : {},
                         'testing' : {}}

        for train_or_test_folder in os.listdir(self.data_folder_path) :
            train_or_test_folder_path = os.path.join(self.data_folder_path, train_or_test_folder)
            
            if os.path.isdir(train_or_test_folder_path) :

                for patient_folder in os.listdir(train_or_test_folder_path) :
                    patient_folder_path = os.path.join(train_or_test_folder_path, patient_folder)

                    if os.path.isdir(patient_folder_path) :

                        infos = {}
                        patient_id = patient_folder.lstrip('patient')

                        #Get the patient informations
                        for line in open(os.path.join(patient_folder_path, 'Info.cfg')):
                                label, value = line.split(':')
                                infos[label] = value.rstrip('\n').lstrip(' ')
                        
                        patients_info[train_or_test_folder][patient_id] = infos
        
        return patients_info

    
    def retrieve_all_files_path(self) : 

        files_paths = {'training' : [],
                        'testing' : []}

        for train_or_test_folder in os.listdir(self.data_folder_path) :
            train_or_test_folder_path = os.path.join(self.data_folder_path, train_or_test_folder)
            
            if os.path.isdir(train_or_test_folder_path) :

                for patient_folder in os.listdir(train_or_test_folder_path) :
                    patient_folder_path = os.path.join(train_or_test_folder_path, patient_folder)

                    if os.path.isdir(patient_folder_path) :

                        for patient_file in glob.glob(os.path.join(patient_folder_path, f'patient???_frame??.nii.gz')):
                            files_paths[train_or_test_folder].append(patient_file)
        
        return files_paths


    def extract_and_preprocess_slices(self) :
        
        subjects = {'training' : [],
                      'testing' : []}

        all_slices = {'training' : {'mri_slices' : [], 'masks' : [], 'patient_id' : []},
                      'testing' : {'mri_slices' : [], 'masks' : [], 'patient_id' : []}}

        files_paths = self.retrieve_all_files_path()
    
        for train_or_test in files_paths.keys() :
            for file_path in files_paths[train_or_test] : 

                base_file = file_path.split('.nii.gz')[0]
                mask_file = base_file + '_gt.nii.gz'
                patient_id = int(file_path.split('/')[7].split('patient')[1])

                img_mri, img_affine, img_header = utils.load_nii(file_path)
                mask = utils.load_nii(mask_file)[0]

                pixel_size = img_header.structarr['pixdim'][:4] 


                 ### PROCESSING LOOP FOR SLICE-BY-SLICE 2D DATA ###################
                cropped_volume_mask, cropped_volume_img = utils.crop_slice_zone_of_interest(mask, img_mri, margin = 10)     
                
                volume_img_normalized = utils.normalize_extreme_values(cropped_volume_img)
                volume_mask_normalized = utils.normalize_extreme_values(cropped_volume_mask)

                volume_img = F.interpolate(volume_img_normalized.unsqueeze(0).permute(3, 0, 1, 2), size = (128, 128))
                volume_mask = F.interpolate(volume_mask_normalized.unsqueeze(0).permute(3, 0, 1, 2), size = (128, 128))

                for slice_index in range(volume_img.shape[0]) : 

                    mri_slice = volume_img[slice_index:slice_index+1, ...]
                    mask_slice = volume_mask[slice_index:slice_index+1, ...]

                    if not np.array(mask_slice == torch.zeros((1, 1, 128, 128))).all() : 

                        all_slices[train_or_test]['mri_slices'].append(mri_slice)
                        all_slices[train_or_test]['masks'].append(mask_slice)
                        all_slices[train_or_test]['patient_id'].append(patient_id)
                            
                        patient = tio.Subject(mri_slice = tio.ScalarImage(tensor = mri_slice), 
                                                mask = tio.LabelMap(tensor = mask_slice))

                        subjects[train_or_test].append(patient)
                        
        return subjects, all_slices

class Partially_Supervised_Loaders() :

    def __init__(self, dataset, all_slices, subjects, cfg) :
        self.dataset = dataset
        self.all_slices = all_slices
        self.subjects = subjects
        self.cfg = cfg

        self.patients_groups_ids = self.get_patients_ids_per_group()
        self.slices_per_groups = self.get_slices_per_groups()

    
    def get_patients_ids_per_group(self) : 

        patients_infos = self.dataset.retrieve_patients_infos()['training']
        patients_groups_ids = {'MINF' : [],
                                'NOR' : [],
                                'RV' : [],
                                'DCM' : [],
                                'HCM' : []}


        for patient_id in patients_infos.keys() :

            patient_info = patients_infos[patient_id]
            patients_groups_ids[patient_info['Group']].append(patient_id)

        return patients_groups_ids



    def get_slices_per_groups(self):

        groups = ['MINF', 'NOR', 'RV', 'DCM', 'HCM']
        slices_per_groups = {'mri_slices' : {'MINF' : [], 'NOR' : [], 'RV' : [], 'DCM' : [], 'HCM' : []},
                            'masks' : {'MINF' : [], 'NOR' : [], 'RV' : [], 'DCM' : [], 'HCM' : []}}

        for group in groups :
            for patient_id in self.patients_groups_ids[group] :

                patient_mris = []
                patient_masks = []
                
                patient_id = int(patient_id)
                slices_idx = (torch.tensor(self.all_slices['training']['patient_id']) == patient_id).nonzero()

                slices = torch.stack(self.all_slices['training']['mri_slices'])[slices_idx, 0, 0, ...]
                masks = torch.stack(self.all_slices['training']['masks'])[slices_idx, 0, 0, ...]
                
                for img_idx in range(slices.shape[0]) :

                    patient_mris.append(slices[img_idx:img_idx+1, ...])
                    patient_masks.append(masks[img_idx:img_idx+1, ...])

                slices_per_groups['mri_slices'][group].append(patient_mris)
                slices_per_groups['masks'][group].append(patient_masks)

        return slices_per_groups


    def build_subjects_list(self) : 

        subjects = []
        groups = ['MINF', 'NOR', 'RV', 'DCM', 'HCM']

        patients_ids_tracking = {key : [i for i in range(20)] for key in groups}

        # Randomly choose n volumes
        if self.cfg.data.num_patients < len(groups) :
            for nb_patient in range(self.cfg.data.num_patients) : 

                group = groups.pop(random.randint(0, len(groups) - 1))
                random_patient_id = random.randint(0, 19)

                patient_mri_slices = self.slices_per_groups['mri_slices'][group][random_patient_id]
                patient_mask_slices = self.slices_per_groups['masks'][group][random_patient_id]

                for slice_idx in range(len(patient_mri_slices)) :

                    mri_slice = patient_mri_slices[slice_idx]
                    mask_slice = patient_mask_slices[slice_idx]

                    patient = tio.Subject(mri_slice = tio.ScalarImage(tensor = mri_slice), 
                                          mask = tio.LabelMap(tensor = mask_slice))

                    subjects.append(patient)

        
        # Pick the same number of volumes for each group
        else : 

            nb_patients_per_group = self.cfg.data.num_patients // len(groups)

            for group in groups :
                for nb_patient in range(nb_patients_per_group) :

                    random_patient_id = patients_ids_tracking[group].pop(random.randint(0, len(patients_ids_tracking[group]) - 1))

                    patient_mri_slices = self.slices_per_groups['mri_slices'][group][random_patient_id]
                    patient_mask_slices = self.slices_per_groups['masks'][group][random_patient_id]

                    for slice_idx in range(len(patient_mri_slices)) :

                        mri_slice = patient_mri_slices[slice_idx]
                        mask_slice = patient_mask_slices[slice_idx]

                        patient = tio.Subject(mri_slice = tio.ScalarImage(tensor = mri_slice), 
                                            mask = tio.LabelMap(tensor = mask_slice))

                        subjects.append(patient)
        
        return subjects

    
    def build_loaders(self) : 

        ########### Preprocessing ###########

        subjects_training = self.build_subjects_list()
        num_subjects = len(subjects_training)

        num_training_subjects = int(self.cfg.data.training_split_ratio * num_subjects)
        num_validation_subjects = num_subjects - num_training_subjects
        num_split_subjects = num_training_subjects, num_validation_subjects

        training_subjects, validation_subjects = torch.utils.data.random_split(subjects_training, num_split_subjects)
        testing_subjects = self.subjects['testing']

        ########### Building Datasets ###########

        # Training Dataset
        cfg_transform = self.cfg.data.training_transform
        training_transform = tio.Compose([
                            tio.RandomNoise(std = cfg_transform.random_noise_std, 
                                            p = cfg_transform.random_noise_p),
                            tio.RandomBlur(std = cfg_transform.random_blur_std, 
                                           p = cfg_transform.random_blur_p),
                            tio.RandomGamma(log_gamma = cfg_transform.log_gamma, 
                                            p = cfg_transform.random_gamma_p),
                            tio.RandomBiasField(coefficients = cfg_transform.random_field_coef, 
                                                p = cfg_transform.random_field_p),
                            transforms.RandomInvert(p = cfg_transform.random_invert_p)    
                            ])
        training_dataset = CustomDataset_Supervised(
            training_subjects, 
            transforms = training_transform)

        # Validation Dataset
        validation_dataset = CustomDataset_Supervised(
            validation_subjects, 
            )

        # Testing Dataset
        testing_dataset = CustomDataset_Supervised(testing_subjects)

        ########### Building Loaders ###########

        training_loader = torch.utils.data.DataLoader(
            training_dataset,
            batch_size = self.cfg.data.batch_size,
            shuffle=True,
        )

        validation_loader = torch.utils.data.DataLoader(
            validation_dataset,
            batch_size = self.cfg.data.batch_size,
            shuffle = True,
        )

        testing_loader = torch.utils.data.DataLoader(
            testing_dataset,
            batch_size = self.cfg.data.batch_size,
            shuffle = True,
        )


        return training_loader, validation_loader, testing_loader


    def build_loaders_for_CL_pretraining(self) : 

        ########### Preprocessing ###########

        subjects = []
        all_mri_slices = self.all_slices['training']['mri_slices']

        for mri_index, mri_slice in enumerate(all_mri_slices) : 

            view_1 = mri_slice.clone().detach()
            view_2 = mri_slice.clone().detach()
            
            patient = tio.Subject(mri_slice_view_1 = tio.ScalarImage(tensor = view_1), 
                                  mri_slice_view_2 = tio.ScalarImage(tensor = view_2))

            subjects.append(patient)

        num_subjects = len(subjects)

        num_training_subjects = int(self.cfg.data.training_split_ratio * num_subjects)
        num_validation_subjects = num_subjects - num_training_subjects
        num_split_subjects = num_training_subjects, num_validation_subjects

        training_subjects, validation_subjects = torch.utils.data.random_split(subjects, num_split_subjects)

        ########### Building Datasets ###########

        # Training Dataset
        cfg_transform = self.cfg.data.training_transform
        training_transform = tio.Compose([
                            tio.RandomNoise(std = cfg_transform.random_noise_std, 
                                            p = cfg_transform.random_noise_p),
                            tio.RandomBlur(std = cfg_transform.random_blur_std, 
                                           p = cfg_transform.random_blur_p),
                            tio.RandomGamma(log_gamma = cfg_transform.log_gamma, 
                                            p = cfg_transform.random_gamma_p),
                            tio.RandomBiasField(coefficients = cfg_transform.random_field_coef, 
                                                p = cfg_transform.random_field_p),
                            transforms.RandomInvert(p = cfg_transform.random_invert_p)    
                            ])
        training_dataset = CustomDataset_CL(
            training_subjects, 
            transforms = training_transform)

        # Validation Dataset
        validation_dataset = CustomDataset_CL(
            validation_subjects)
        

        ########### Building Loaders ###########

        training_loader = torch.utils.data.DataLoader(
            training_dataset,
            batch_size = self.cfg.data.batch_size_CL,
            shuffle=True,
            # drop_last = True
        )

        validation_loader = torch.utils.data.DataLoader(
            validation_dataset,
            batch_size = self.cfg.data.batch_size_CL,
            shuffle = True,
        )

        return training_loader, validation_loader

    def build_test_volume_loader(self) :
      
        subjects_volume_test = []

        for patient_id in range(101, 151) :
            slices_idx = (torch.tensor(self.all_slices['testing']['patient_id']) == patient_id).nonzero()

            slices = torch.stack(self.all_slices['training']['mri_slices'])[slices_idx, 0, 0, ...]
            masks = torch.stack(self.all_slices['training']['masks'])[slices_idx, 0, 0, ...]

            patient = tio.Subject(mri_slice = tio.ScalarImage(tensor = slices),
                                mask = tio.ScalarImage(tensor = masks))

            subjects_volume_test.append(patient)
            
        testing_dataset_volume = CustomDataset_Supervised(subjects_volume_test)

        testing_loader_volume = torch.utils.data.DataLoader(
                    testing_dataset_volume,
                    batch_size = 1,
                    shuffle = False,
                )

        return testing_loader_volume

class CustomDataset_Supervised(Dataset):
    def __init__(self,
                 subjects,
                 transforms = None):
        
        self.subjects = subjects
        self.transform = transforms

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):

        subject = self.subjects[idx]
        mri_slice = subject['mri_slice'].data
        mask = subject['mask'].data

        if self.transform is not None:
            mri_slice = self.transform(mri_slice)

        return mri_slice, mask
        

class CustomDataset_CL(Dataset):
    def __init__(self,
                 subjects,
                 transforms = None):
        
        self.subjects = subjects
        self.transform = transforms

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):

        subject = self.subjects[idx]
        view_1 = subject['mri_slice_view_1'].data
        view_2 = subject['mri_slice_view_2'].data

        if self.transform is not None:
            view_1 = self.transform(view_1)
            view_2 = self.transform(view_2)

        return view_1, view_2