import torch
import os
import json
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from unet_utils import Unet_Utils
from model import UNet
from dataloader import Cell_data

class Experiment:
    def __init__(self, filename):
        #create a dict to store process info
        self.info = {}
        self.info['input_settings'] = {}
        self.info['training'] = {}
        root_dir = os.getcwd()
        #initialise util
        util = Unet_Utils()
        #parse main file
        (settings, model_params, dataset_params) = util.parse_config_file(filename)
        self.info['input_settings']['settings'] = settings
        self.info['input_settings']['model_params'] = model_params
        self.info['input_settings']['dataset_params'] = dataset_params
        #parse settings
        (self.epoch_n, self.image_size, use_gpu) = util.parse_experiment_settings(settings)
        self.device = util.find_device(use_gpu)
        #parse model params
        (use_existing_model, existing_model_filename, self.final_model_filename, self.experiment_info_filename, self.criterion, opt_settings) = util.parse_model_settings(model_params)
        self.criterion.to(self.device)
        self.model = UNet()
        if use_existing_model:
            existing_model_filename = os.path.join(root_dir, existing_model_filename)
            self.model.load_state_dict(torch.load(existing_model_filename))
        self.model.to(self.device)
        self.optimizer = util.get_optimizer(self.model.parameters(), opt_settings)
        self.final_model_filename = os.path.join(root_dir, self.final_model_filename)
        self.experiment_info_filename = os.path.join(root_dir, self.experiment_info_filename)
        #parse dataset settings
        (images_directory, train_test_split, self.batch_size, augment_data) = util.parse_dataset_settings(dataset_params)   
        data_dir = os.path.join(root_dir, images_directory)
        self.trainset = Cell_data(data_dir = data_dir, size = self.image_size, augment_data=augment_data, train_test_split=train_test_split)
        self.trainloader = DataLoader(self.trainset, batch_size = self.batch_size, shuffle=True)
        self.testset = Cell_data(data_dir = data_dir, size = self.image_size, train = False, augment_data=False, train_test_split=train_test_split)
        self.testloader = DataLoader(self.testset, batch_size = self.batch_size)
    
    def train_model(self):
        for e in range(self.epoch_n):
            epoch_loss = 0
            self.info['training'][e] = {}
            self.info['training'][e]['batch_loss'] = []
            self.model.train()
            for i, data in enumerate(self.trainloader):
                image, label = data
                image = image.unsqueeze(1).to(self.device)
                label = label.long().to(self.device)

                pred = self.model(image)

                crop_x = (label.shape[1] - pred.shape[2]) // 2
                crop_y = (label.shape[2] - pred.shape[3]) // 2

                label = label[:, crop_x: label.shape[1] - crop_x, crop_y: label.shape[2] - crop_y]

                loss = self.criterion(pred, label)
                loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()

                epoch_loss += loss.item()

                batch_loss = loss.item() / self.batch_size
                self.info['training'][e]['batch_loss'].append(batch_loss)
                print('batch %d --- Loss: %.4f' % (i, batch_loss))
            epoch_loss =  epoch_loss / self.trainset.__len__()
            self.info['training'][e]['epoch_loss'] = epoch_loss   
            print('Epoch %d / %d --- Loss: %.4f' % (e + 1, self.epoch_n, epoch_loss))

            torch.save(self.model.state_dict(), 'checkpoint.pt')

            self.model.eval()

            total = 0
            correct = 0
            total_loss = 0

            with torch.no_grad():
                for i, data in enumerate(self.testloader):
                    image, label = data
                    image = image.unsqueeze(1).to(self.device)
                    label = label.long().to(self.device)

                    pred = self.model(image)
                    crop_x = (label.shape[1] - pred.shape[2]) // 2
                    crop_y = (label.shape[2] - pred.shape[3]) // 2

                    label = label[:, crop_x: label.shape[1] - crop_x, crop_y: label.shape[2] - crop_y]

                    loss = self.criterion(pred, label)
                    total_loss += loss.item()

                    _, pred_labels = torch.max(pred, dim=1)

                    total += label.shape[0] * label.shape[1] * label.shape[2]
                    correct += (pred_labels == label).sum().item()

            accuracy = correct / total
            avg_loss = total_loss / self.testset.__len__()
            self.info['training'][e]['accuracy'] = accuracy
            self.info['training'][e]['eval_loss'] = avg_loss
            print('Accuracy: %.4f ---- Loss: %.4f' % (accuracy, avg_loss))

    def evaluate_model(self):
        self.model.eval()
    
        output_masks = []
        output_labels = []

        with torch.no_grad():
            for i in range(self.testset.__len__()):
                image, labels = self.testset.__getitem__(i)

                input_image = image.unsqueeze(0).unsqueeze(0).to(self.device)
                pred = self.model(input_image)

                output_mask = torch.max(pred, dim=1)[1].cpu().squeeze(0).numpy()

                crop_x = (labels.shape[0] - output_mask.shape[0]) // 2
                crop_y = (labels.shape[1] - output_mask.shape[1]) // 2
                labels = labels[crop_x: labels.shape[0] - crop_x, crop_y: labels.shape[1] - crop_y].numpy()

                output_masks.append(output_mask)
                output_labels.append(labels)

        fig, axes = plt.subplots(self.testset.__len__(), 2, figsize = (20, 20))

        for i in range(self.testset.__len__()):
            axes[i, 0].imshow(output_labels[i], cmap='gray')
            axes[i, 0].axis('off')
            axes[i, 1].imshow(output_masks[i], cmap='gray')
            axes[i, 1].axis('off')

    def save_final_model(self):
        torch.save(self.model.state_dict(),self.final_model_filename)

    def save_experiment_info(self):
        with open(self.experiment_info_filename, 'w') as f:
            json.dump(self.info, f, indent=3)
        f.close()    