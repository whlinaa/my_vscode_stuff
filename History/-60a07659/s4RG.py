import os
import zipfile
from pathlib import Path
# %%
def run_and_zip(file, output_name=None, ts=None, variation=None):
    # print(f"jupyter nbconvert --to notebook --execute {file}.ipynb --output {file}_out.ipynb")
    os.system(f"jupyter nbconvert --to notebook --execute {file}.ipynb --output {file}_out.ipynb")
# %%
# run_and_zip('/Users/whlin/Library/CloudStorage/OneDrive-HKUSTConnect/Documents/python/my_notes_py/test_output')



print(str(Path("/hello/world")))

