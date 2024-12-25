from runner import Runner
from common.arguments import get_args
from common.utils import make_env
import numpy as np
import random
import torch
import os
os.environ['SUPPRESS_MA_PROMPT'] = '1'


# python main.py --scenario-name=simple_tag --evaluate-episodes=10

if __name__ == '__main__':
    # get the params
    args = get_args()
    env, args = make_env(args)
    runner = Runner(args, env)
    if args.evaluate:
        returns = runner.evaluate()
        print('Average returns is', returns)
    else:
        runner.run()
