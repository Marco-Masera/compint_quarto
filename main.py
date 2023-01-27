import logging
import argparse
from quarto_agent import QuartoAgent
from tests import TestAgent
from quarto_agent_lib.generate_dataset import generate_dataset_undeterministic, generate_dataset
from quarto_agent_lib.pre_process_dataset import PreProcessDataset
from quarto_agent_lib.train_reward_function import climb

#Get agent playing quarto
def get_agent(game):
    return QuartoAgent(game)
#Run tests
def run_tests():
    TestAgent.run_test(24)

### Methods to train the reward function
def generate_dataset_for_training(deterministic = True, export_fn="dataset/raw/dataset_v0", depth=8, stop_at=None):
    if (deterministic):
        generate_dataset(export_fn=export_fn, depth=8)
    else:
        generate_dataset_undeterministic(export_fn=export_fn, depth=8, stop_at=stop_at)
#Inputs are paths to the datasets to process
def pre_process_dataset(inputs):
    PreProcessDataset.pre_process(inputs)
def train_reward_function():
    climb()

def main():
    run_tests()
    #train_reward_function()
    #generate_dataset_for_training(True)
    #pre_process_dataset(["path/to/raw/dataset"]) #Raw datasets haven't been committed to github due to size

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='count',
                        default=0, help='increase log verbosity')
    parser.add_argument('-d',
                        '--debug',
                        action='store_const',
                        dest='verbose',
                        const=2,
                        help='log debug messages (same as -vv)')
    args = parser.parse_args()

    if args.verbose == 0:
        logging.getLogger().setLevel(level=logging.WARNING)
    elif args.verbose == 1:
        logging.getLogger().setLevel(level=logging.INFO)
    elif args.verbose == 2:
        logging.getLogger().setLevel(level=logging.DEBUG)

    main()