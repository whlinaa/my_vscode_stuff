import os
import zipfile
# %%
def run_and_zip(file, output_name, ts, variation):
    os.system(f"jupyter nbconvert --to notebook --execute {hb_loss_ratio_model}.ipynb --output {hb_loss_ratio_model}_out.ipynb")
    
    
    