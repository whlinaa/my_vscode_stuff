from pathlib import Path
import argparse

def add_prefix_suffix(path, ext='', prefix='', suffix='', ignore_hidden=True, recursive=False):
    # ext: filter by extension 
    # use case: add timestamp to each file
    
    if ignore_hidden:
        pattern = f'[!.]*.{ext}' if ext else '[!.]*.*'
    else:
        pattern = f'*.{ext}' if ext else '*.*'
    
        
    files = sorted(Path(path).expanduser().rglob(pattern)) if recursive else sorted(Path(path).expanduser().glob(pattern))
    
    for file in files:
        # print(file)
        new_name = '_'.join(x for x in [prefix, file.stem, suffix] if x) # we need list comprehension to exclude those empty string input!
        # print(file.with_stem(new_name))
        file.rename(file.with_stem(new_name))


parser = argparse.ArgumentParser(prog='add_prefix_suffix', description='add prefix and/or suffix to the files in a given directory')
parser.add_argument('path', action='store', help='optional positional argument with default')
parser.add_argument('-e', '-ext', action='store', help='change the files with given extension only')
# parser.add_argument('prefix', action='store', help='optional positional argument with default')
# parser.add_argument('suffix', action='store', help='optional positional argument with default')
# parser.add_argument('ignore_hidden', action='store', help='optional positional argument with default')
# parser.add_argument('recursive', action='store', help='optional positional argument with default')



args = parser.parse_args()
print(vars(args))

# path = '~/Desktop/hello/world/dir_1'
# path = test_path / "new_dir"
# path = '~/Desktop/foo'
# create_dir(path)
# add_prefix_suffix(path, prefix='pref', suffix='suff', ignore_hidden=True, recursive=True)
