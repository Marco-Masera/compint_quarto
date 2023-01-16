import logging
import argparse
import random
import quarto
from quarto_agent import QuartoAgent
from tests import *
from train_agent_params import train


def train_agent_params():
    train()

def main():
    #test_against_random_agent()
    #test_against_random_reward()
    #test_single_match()
    #test_cache_vs_no_cache()
    #test_cache_vs_cache()
    train_agent_params()


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