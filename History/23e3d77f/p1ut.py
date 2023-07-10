# %%
import sys
import os

def change_file_names(path, ext='png', prefix=None, suffix=None):
    for file in os.listdir(path):
        if file.endswith('ext'):
            print(file)
            os.rename(os.path.join(path, file), os.path.join(path, f'{prefix}_{file}'))

        # print(file)


# path='~/Downloads'
path = os.path.expanduser('~/Desktop/my_fig_2')
print(path)
change_file_names(path)

# %%
# ! ls -l
os.path.join?
# os.listdir?