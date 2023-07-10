# %%
import os
import numpy as np
import shutil
import webbrowser
import time

# %%
# cwd = r"D:\Academic\Notes of HKCC\HKCC Teaching\SEHH1070\quiz 1\203"  # get current directory
# cwd = r"D:\Academic\Notes of HKCC\HKCC Teaching\SEHH1070\midterm\student papers\203"  # get current directory
# cwd = r"D:\Academic\Notes of HKCC\HKCC Teaching\SEHH2042\SEHH2042_21Spring\midterm\202\202\Version B"  # get current directory
# cwd = r"D:\Academic\Notes of HKCC\HKCC Teaching\SEHH2241\SEHH2241_21_Spring\midterm\student paper"  # get current directory
# cwd=r"D:\Academic\Notes of HKCC\HKCC Teaching\SEHH2042\SEHH2042_21Spring\midterm\202\201\Version B"
# cwd=r"D:\Academic\Notes of HKCC\HKCC Teaching\SEHH2042\SEHH2042_21Spring\midterm\205\205_Q1_Q2\Version B"
# cwd=r"D:\Academic\Notes of HKCC\HKCC Teaching\SEHH2042\SEHH2042_21Spring\midterm\205\205_Q1_Q2\Version A"
# cwd=r"D:\Academic\Notes of HKCC\HKCC Teaching\SEHH1070\ASM\student work\203"

# 204
# cwd=r"D:\Academic\Notes of HKCC\HKCC Teaching\SEHH2042\SEHH2042_21Spring\midterm\204\204_Q1_Q2_Q3_Q4\Version A"
# cwd=r"D:\Academic\Notes of HKCC\HKCC Teaching\SEHH2042\SEHH2042_21Spring\midterm\204\204_Q1_Q2_Q3_Q4\Version B"

# 203
# cwd=r"D:\Academic\Notes of HKCC\HKCC Teaching\SEHH2042\SEHH2042_21Spring\midterm\203\203_Q4_Q1_Q2\Version A"
# cwd=r"D:\Academic\Notes of HKCC\HKCC Teaching\SEHH2042\SEHH2042_21Spring\midterm\203\203_Q4_Q1_Q2\Version B"

# cwd=r"D:\Academic\Notes of HKCC\HKCC Teaching\SEHH2241\SEHH2241_21_Spring\A2\student work"
# cwd=r"C:\Users\whlin\OneDrive - College of Professional and Continuing Education (CPCE)\My files\!!! SEHH1070 Exam\201"
# cwd=r"C:\Users\whlin\OneDrive - College of Professional and Continuing Education (CPCE)\My files\!!! SEHH1070 Exam\202"
# cwd=r"C:\Users\whlin\OneDrive - College of Professional and Continuing Education (CPCE)\My files\!!! SEHH1070 Exam\203"

cwd=r'/Users/whlin/Library/CloudStorage/OneDrive-HKUSTConnect/Academic/Notes of HKCC/HKCC Teaching/SEHH2239/exam things/202'

# %%
for file in os.listdir(cwd):
	print(file)
# %%

names=[]
for file in os.listdir(cwd):
	if file != '.DS_Store':
	# for item in os.listdir(f'{cwd}\{file}'):
		for item in os.listdir(f'{cwd}/{file}'):
			# names.append(f'{file}\{item}')
			names.append(f'{file}/{item}')
		# print(item)——
		# os.startfile(f'{cwd}\{file}\{item}')
		# webbrowser.open_new(f'{cwd}\{file}\{item}')
		# input("Press Enter to continue...")

start=int(input("enter start index(start from 1): "))
end=int(input("enter end index: "))


for i in range(start - 1, end):
	# print(i)
	# print(name)
	os.startfile(f'{cwd}\{names[i]}')
	time.sleep(0.8)

# for name in names:
# 	# print(name)
# 	os.startfile(f'{cwd}\{name}')
# 	time.sleep(0.1)


# %%
# if each file is not directory
names=[]
for file in os.listdir(cwd):
		names.append(f'{file}')
		# print(item)——
		# os.startfile(f'{cwd}\{file}\{item}')
		# webbrowser.open_new(f'{cwd}\{file}\{item}')
		# input("Press Enter to continue...")
names
# %%

start=int(input("enter start index(start from 1): "))
end=int(input("enter end index: "))

for i in range(start - 1, end):
	# print(i)
	# print(name)
	os.startfile(f'{cwd}\{names[i]}')
	time.sleep(0.7)

# for name in names:
# 	# print(name)
# 	os.startfile(f'{cwd}\{name}')
# 	time.sleep(0.1)

# %%
def most_frequent(t):
    # M0: use t.count (https://bit.ly/2QHgL7q)
    # return max(t, key = t.count)

    # M1: M0, with set() to improve efficiency
    # return max(set(t), key = t.count)

    # M2: Counter
    # freq_count=Counter(t)
    # return max(freq_count, key= lambda key: freq_count[key])

    # M3: use the most_common method of Counter
    return Counter(t).most_common(1)[0][0]

t=[1,2,1,1,2,1,1,1,-1]
most_frequent(t)
