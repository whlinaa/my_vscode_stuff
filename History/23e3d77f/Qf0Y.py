# %%
import sys
import os

def change_file_names(path, ext='png', prefix=None, suffix=None):
    for file in os.listdir(path):
        # print(type(file))
        if file.endswith(ext):
            new_name = file
            if prefix:
                new_name = prefix + '_' + new_name
            if suffix:
                new_name = new_name + '_' + suffix

            os.rename(os.path.join(path, file), os.path.join(path, new_name))

        # print(file)


# path='~/Downloads'
path = os.path.expanduser('~/Desktop/my_fig_2')
print(path)
change_file_names(path)

# %%
# ! ls -l
os.path.join?
# os.listdir?