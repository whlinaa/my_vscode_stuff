
# %%
import time
# while True:
#     x = input("input: ")
#     print(x)


print("hello", end=' ', flush=False) # this statement will be printed at the same time as the next world statement...
time.sleep(2)
print("world", end=" ")
time.sleep(2)
print("again", end=" ")
time.sleep(2)
