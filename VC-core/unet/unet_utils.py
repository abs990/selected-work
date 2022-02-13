from typing import Dict, List, Tuple
from cv2 import norm
from matplotlib.colors import Normalize
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import matplotlib.pyplot as plt
import os
import json
from model import UNet
from dataloader import Cell_data

class Unet_Utils:
    def key_check(self, dict, key_list) -> None:
        assert all (x in dict for x in key_list)

    def get_criterion(self, criterion):
        if criterion == 'CrossEntropyLoss':
            return nn.CrossEntropyLoss()

    def get_optimizer(self,model_params, optimizer_dict):
        name_key = 'name'
        assert name_key in optimizer_dict
        if optimizer_dict[name_key] == 'Adam':
            opt_param_keys = ['lr','weight_decay']
            self.key_check(optimizer_dict, opt_param_keys)
            return optim.Adam(model_params, lr=optimizer_dict['lr'], weight_decay=optimizer_dict['weight_decay'])

    def find_device(self, use_gpu: bool) -> str:
        if use_gpu and torch.cuda.is_available():
            return 'cuda:0'
        else:
            return 'cpu'

    def parse_experiment_settings(self,settings_dict: Dict) -> Tuple[int, int, bool]:
        settings_keys = ['epoch_n','image_size','use_gpu']
        self.key_check(settings_dict, settings_keys)
        exp_settings = []
        for key in settings_keys:
            value = settings_dict[key]
            if key == 'epoch_n':
                assert isinstance(value, int)
                assert value > 0
            elif key == 'image_size':
                assert isinstance(value, int)
                assert value == 572
            elif key == 'use_gpu':
                assert isinstance(value, bool)        
            exp_settings.append(value)
        return tuple(exp_settings)

    def parse_model_settings(self,model_dict: Dict) -> Tuple:
        model_keys = ['use_existing_model','existing_model_filename','final_model_filename','experiment_info_filename','criterion','optimizer']
        self.key_check(model_dict, model_keys)
        model_params = []
        for key in model_keys:
            value = model_dict[key]
            if key == 'use_existing_model':
                assert isinstance(value, bool)
            elif key in ('existing_model_filename','final_model_filename','experiment_info_filename'):
                assert isinstance(value, str)
            elif key == 'criterion':
                assert isinstance(value, str)
                value = self.get_criterion(value)
            elif key == 'optimizer':
                assert isinstance(value, Dict)
            model_params.append(value)
        return tuple(model_params)               

    def parse_dataset_settings(self, dataset_dict: Dict) -> Tuple:
        dataset_keys = ['images_directory', 'train_test_split', 'batch_size','augment_data']
        self.key_check(dataset_dict, dataset_keys)
        dataset_params = []
        for key in dataset_keys:
            value = dataset_dict[key]
            if key == 'images_directory':
                assert isinstance(value, str)
            elif key == 'train_test_split':
                assert isinstance(value, float)
                assert value < 1 and value > 0
            elif key == 'batch_size':
                assert isinstance(value, int)
                assert value > 1
            elif key == 'augment_data':
                assert isinstance(value, bool)        
            dataset_params.append(value)
        return tuple(dataset_params)

    def parse_config_file(self, filename: str) -> Tuple[Dict, Dict, Dict]:                
        self.key_list = ['settings','model','dataset']
        with open(filename,"r") as stream:
            try:
                file_entries = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        stream.close()
        assert all (x in file_entries for x in self.key_list)
        settings = file_entries['settings']
        model_params = file_entries['model']
        dataset_params = file_entries['dataset']
        return (settings, model_params, dataset_params)

    def visualize_experiment_plots(self, filename:str) -> None:
        def extract_train_test_losses(exp_dict: Dict) -> Tuple:
            train_loss = {}
            test_loss = {}
            training_key = 'training'
            train_loss_key = 'epoch_loss'
            test_loss_key = 'eval_loss'
            for key in exp_dict[training_key].keys():
                train_loss[key] = exp_dict[training_key][key][train_loss_key]
                test_loss[key] = exp_dict[training_key][key][test_loss_key]
            return (train_loss, test_loss)

        root_dir = os.getcwd()
        file_dir = os.path.join(root_dir, filename)
        with open(file_dir) as json_file:
            exp_info = json.load(json_file)
        json_file.close()    
        (train_err, test_err) = extract_train_test_losses(exp_info)
        plt.figure
        plt.plot(train_err.keys(), train_err.values(), '-bo')
        plt.plot(test_err.keys(), test_err.values(), '-ro')
        plt.ylabel('Loss values')
        plt.legend(['Training error','Test error'])
        plt.title('Plot of train and test losses')
        plt.xlabel('Epoch number')
        plt.show()

    def evaluate_custom_model(self, filename: str) -> None:
        root_dir = os.getcwd()
        file_dir = os.path.join(root_dir, filename)
        model = UNet()
        model.load_state_dict(torch.load(file_dir))
        model.eval()
        output_masks = []
        output_labels = []
        device = 'cpu'
        testset = Cell_data(data_dir = 'data/cells', size = 572, train = False, augment_data=False, train_test_split=0.8)
        with torch.no_grad():
            for i in range(testset.__len__()):
                image, labels = testset.__getitem__(i)

                input_image = image.unsqueeze(0).unsqueeze(0).to(device)
                pred = model(input_image)

                #output_mask = torch.max(pred, dim=1)[1].cpu().squeeze(0).numpy()
                output_mask = torch.max(pred, dim=1)[1].cpu().squeeze(0)

                crop_x = (labels.shape[0] - output_mask.shape[0]) // 2
                crop_y = (labels.shape[1] - output_mask.shape[1]) // 2
                labels = labels[crop_x: labels.shape[0] - crop_x, crop_y: labels.shape[1] - crop_y].numpy()

                print(output_mask)

                output_masks.append(output_mask)
                output_labels.append(labels)

        fig, axes = plt.subplots(testset.__len__(), 2, figsize = (20, 20))

        for i in range(testset.__len__()):
            axes[i, 0].imshow(output_labels[i])
            axes[i, 0].axis('off')
            axes[i, 1].imshow(output_masks[i], interpolation='nearest',
               vmin=0, vmax=1)
            axes[i, 1].axis('off')

        plt.show()                    