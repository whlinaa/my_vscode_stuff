# %%
import numpy as np
import argparse
# %%
my_parser = argparse.ArgumentParser(prog='my_program', description='my testing program')
my_group = my_parser.add_mutually_exclusive_group(required=False)
my_group.add_argument('--min', action='store_true', default=True)
my_group.add_argument('--max', action='store_true', default=False)

args = my_parser.parse_args()
# %%
def find_extreme(t):
    if args.min:
        return min(t)
    else:
        return min(t)