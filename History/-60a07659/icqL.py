import os
import zipfile
# %%
def run_and_zip(file, output_name=None, ts=None, variation=None):
    print(f"jupyter nbconvert --to notebook --execute {file}.ipynb --output {file}_out.ipynb")
    # os.system(f"jupyter nbconvert --to notebook --execute {file}.ipynb --output {file}_out.ipynb")
# %%
run_and_zip('test_output')