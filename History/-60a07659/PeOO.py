import os
import zipfile
from pathlib import Path
# %%
def run_and_zip(file, output_name=None, ts=None, variation=None):
    file = Path(file)
    # print(f"jupyter nbconvert --to notebook --execute {file}.ipynb --output {file}_out.ipynb")
    os.system(f"jupyter nbconvert --to notebook --execute {str(file)} --output {str(file.with_stem('_'.join([file.stem, "out"])))}"
# %%
run_and_zip('/Users/whlin/Library/CloudStorage/OneDrive-HKUSTConnect/Documents/python/my_notes_py/test_output.ipynb')



# p = Path("/hello/world/hello.txt")
# print(p)

# print(p.with_stem('_'.join([p.stem, "out"])))


# {str(file.with_stem('_'.join([file.stem, "out"])))}

