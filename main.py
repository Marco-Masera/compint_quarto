import logging
import argparse
import random
import quarto
from quarto_agent import QuartoAgent
from tests import TestAgent
from train_agent_params import train
from train_rl_agent import TrainRLAgent

def run_tests():
    TestAgent.run_test(5)

def train_agent_params():
    train()

def train_rl_agent():
    TrainRLAgent.train(30)

def main():
    run_tests()
    #train_agent_params()
    #train_agent_params

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