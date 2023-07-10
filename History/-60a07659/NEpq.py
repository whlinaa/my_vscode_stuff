import os
import zipfile
# %%
def run_and_zip(file, output_name, ts, variation):
    os.system(f"jupyter nbconvert --to notebook --execute {}}.ipynb --output {}}_out.ipynb")
    
    
    