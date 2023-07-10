# %%
import os
""" Append each document with an given id.
	EX: 1234556A_ASM => 1_1234556A_ASM if the document has id 1 
"""
data_path=r"D:\Documents\python\data\SEHH2042_stuID+gp.txt"
doc_path=r'D:\Academic\Notes of HKCC\HKCC Teaching\SEHH2042\SEHH2042_21Spring\project\student work\204\p2p'
os.chdir(doc_path) # change current directory to where the files to be renamed is
# os.getcwd()

d=dict() # contains the mapping from student id to group id 
fin=open(data_path)
for line in fin:
	id, gp=line.strip().split('\t')
	d[id]=gp
print(d)

for doc in os.listdir(os.getcwd()):
	if ".docx" in doc:
		new=doc[:9].upper()
		if new in d.keys():
			new= d[new]+'__'+new+".docx"
			print(new)
		else:
			print('error: the name is not in DB')
		os.rename(doc,new)
	







