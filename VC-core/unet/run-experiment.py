from experiment import Experiment
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--config", type=str,  default='experment-parameters/experiment-demo.yml')
args = parser.parse_args()
filename = args.config
exp = Experiment(filename=filename)
exp.train_model()
exp.evaluate_model()
exp.save_final_model()
exp.save_experiment_info()