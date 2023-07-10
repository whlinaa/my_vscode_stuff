# %%
import sys
import os

def change_file_names(path, ext='png', prefix=None, suffix=None):
    for file in os.listdir(path):
        # print(type(file))
        if file.endswith(ext):
            new_name = file
            if prefix:
                # new_name = prefix + '_' + new_name
                new_name = f'{prefix}_{new_name}'
            if suffix:
                # new_name = new_name + '_' + suffix
                new_name = f'{new_name}_{suffix}'
            print(new_name)
            os.rename(os.path.join(path, file), os.path.join(path, new_name))

        # print(file)


# path='~/Downloads'
path = os.path.expanduser('~/Desktop/my_fig_2')
print(path)
change_file_names(path, prefix=1, suffix=5)

# %%
# ! ls -l
os.path.join?
# os.listdir?
# %%
from pathlib import Path

def append_id(filename):
  p = Path(filename)
  print(p.stem)
  print(p.suffix)
#   return "{0}_{2}{1}".format(p.stem, p.suffix, 1234)


append_id('~/hello/world/file.txt')
# %%
p = Path(p.expanduser('~/Desktop/test_folder'))
print(p.name)
print(p.parent)
print(p.root)
print(p.suffix)
print(p.name)

for x in p.iterdir():
    print(x)