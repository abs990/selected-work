from sklearn.model_selection import train_test_split
from dataloader import Cell_data
import cv2
import matplotlib.pyplot as plt
from model import UNet
from unet_utils import Unet_Utils
from experiment import Experiment

train_test_split = 0.8
cd = Cell_data(data_dir='data/cells', size=572, train_test_split=train_test_split)
assert cd.__len__() == 30
cd = Cell_data(data_dir='data/cells', size=572, train_test_split=train_test_split, train=False)
assert cd.__len__() == 8

image, label = cd.__getitem__(0)
model = UNet().to('cpu')
input = image.view(1,1,572,572)
pred = model(input)
assert list(pred.size()) == [1, 2, 388, 388]

util = Unet_Utils()
filename = 'experiment-parameters/experiment-demo.yml'
(s, m, d) = util.parse_config_file(filename)
print(s, m, d)
(s_e, s_is, s_gpu) = util.parse_experiment_settings(s)
print(s_e, s_is, s_gpu)
(m1, m2, m3, m4, m5, m6) = util.parse_model_settings(m)
print(m1, m2, m3, m4, m5, m6)
(d1, d2, d3, d4) = util.parse_dataset_settings(d)
print(d1, d2, d3, d4)

exp = Experiment(filename=filename)
exp.train_model()
exp.evaluate_model()
exp.save_final_model()
exp.save_experiment_info()