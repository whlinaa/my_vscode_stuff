#!/usr/bin/env python
# %%
from pathlib import Path
import argparse
import datetime

def get_timestamp():
    # return datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def add_prefix_suffix(path, ext='', prefix='', suffix='', ignore_hidden=True, recursive=False, timestamp=False):
    # ext: filter by extension 
    # use case: add timestamp to each file
    
    if ignore_hidden:
        pattern = f'[!.]*.{ext}' if ext else '[!.]*.*'
    else:
        pattern = f'*.{ext}' if ext else '*.*'
    
    files = sorted(Path(path).expanduser().rglob(pattern)) if recursive else sorted(Path(path).expanduser().glob(pattern))
    
    # print(files)
    ts = get_timestamp()
    
    for file in files:
        # print(file)
        new_name = '_'.join(x for x in [prefix, file.stem, suffix] if x) # we need list comprehension to exclude those empty string input!
        if timestamp == 'prefix':
            new_name = '_'.join([ts, new_name])
        elif timestamp == 'suffix':
            new_name = '_'.join([new_name, ts])
        # print(file.with_stem(new_name))
        file.rename(file.with_stem(new_name))

# print("helo")
# %%
# parse arguments
parser = argparse.ArgumentParser(prog='add_prefix_suffix', description='add prefix and/or suffix to the files in a given directory')
parser.add_argument('path', action='store', help='location of the file.s defaults to None', nargs='?', default=None)
parser.add_argument('-e', '--ext', action='store', help='change the files with given extension only', default='')
parser.add_argument('-p', '--prefix', action='store', help='add the prefix to all files', default='')
parser.add_argument('-s', '--suffix', action='store', help='add the suffix to all files', default='')
parser.add_argument('-i', '--ignore_hidden', action='store_true', help='ignore hidden files', default=True)
parser.add_argument('-r', '--recursive', action='store_true', help='rename recursively', default=False)
parser.add_argument('-t', '--timestamp', action='store', help='add timestamp as prefix or suffix', choices=['prefix', 'suffix'], default=False)

args = parser.parse_args()

print(f"{vars(args) = }")

args = vars(args)

# print(args.recursive)

# path = '~/Desktop/hello/world/dir_1'
# path = test_path / "new_dir"
# path = '~/Desktop/foo'
# create_dir(path)

if args['path']: 
    # print("enter")
    add_prefix_suffix(
        path=args['path'], 
        prefix=args['prefix'], 
        suffix=args['suffix'], 
        ignore_hidden=args['ignore_hidden'], 
        recursive=args['recursive'], 
        timestamp=args['timestamp']
        )