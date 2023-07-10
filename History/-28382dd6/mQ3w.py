# %%
import numpy as np
import argparse
# %%
my_parser = argparse.ArgumentParser(prog='my_program', description='my testing program')
my_parser.add_argument('-i', '--input', type=list, default=[1,2,3,4,5])
my_parser.add_argument('--max', type=bool, default=True)


# my_group = my_parser.add_mutually_exclusive_group(required=False)
# my_group.add_argument('--min', action='store_true', default=True)
# my_group.add_argument('--max', action='store_true', default=False)

args = my_parser.parse_args()
# %%
def find_extreme(t):
    if args.min:
        return min(t)
    else:
        return max(t)

print(vars(args))
print(find_extreme(args.input))