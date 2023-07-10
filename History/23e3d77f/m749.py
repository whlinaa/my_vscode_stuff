# %%
import sys
import os

def change_file_names(path, ext='png', prefix=None, suffix=None):
    for file in os.listdir(path):
        os.rename(path+'/'+file, f'{path}+'/'+prefix_{file}')
        # print(file)


# path='~/Downloads'
path = os.path.expanduser('~/Desktop/my_fig2')
print(path)
change_file_names(path)

# %%
# ! ls -l

os.listdir?