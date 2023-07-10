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
# /Users/whlin/Desktop/test_folder/pic1.png
# p = Path('~/Desktop/test_folder')
p = Path('/Users/whlin/Desktop/test_folder')
# p = Path(p.expanduser('~/Desktop/test_folder'))
print(p.name)
print(p.parent)
print(p.root)
print(p.suffix)

# for x in p.iterdir():
for x in p.glob('*.png'):
    print(x)
# %%
# os.getcwd()
# os.chdir('..')
# os.getcwd()
os.chdir('/Users/whlin/Desktop/test_folder')
# %%
os.getcwd()
os.rename()
# %%
cwd = os.getcwd()
# print(os.getcwd())
# os.chdir('/Users/whlin/Library/CloudStorage/OneDrive-HKUSTConnect/Documents/python/my_modules/file_management/textfiles')
fin= open("score.txt")
# in score.txt, we have three columns, each separated by \t
# 17101349a Average 20
# 18039154a Average 20
# 18044059a Minimum 20
# 18062182a Minimum 20
# 18120927a Minimum 20
# 18124453a Minimum 20
# 18151049a Average 20
# 18157969a Average 20
data={}
for line in fin:
    # print(line.strip().split('\t'))
    id,version,score=line.strip().split('\t')
    data[id]=version,score
print(data)

for item in os.listdir(cwd): # the file is originally named with id 
    if(os.path.isdir(item)): # if it is a directory
        newFileName=f'{data[item][0]}_{data[item][1]}_{item}' # Average_20_17101349a
        # print(newFileName)
        print(item)
        print(newFileName)
        os.rename(item,newFileName)

# %%
path = '/tmp/abc/efg/test.txt'
print(os.path.basename(path)) #  test.txt
print(os.path.dirname(path)) #  /tmp/abc/efg
os.path.split(path) #  ('/tmp/abc/efg', 'test.txt')
# print(os.path.splittext(path))