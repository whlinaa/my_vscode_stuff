# %%

for i in range(5, 9):
    print(i)
    n = 10**i

    t = list(range(n))
    s = set(range(n))

    # print(t)
    # print(s)

    %time print(-10 in t)
    %time print(-10 in s)

# 10**6 micro sec = 1 sec

# %%

# 10**7 => 10**8
n = 10**7 # size of a array
t = list(range(n))
s = set(range(n)) # set is a data structure that uses the hashing technique


%time -10 in t # linear search 
%time -10 in s # hashing


# hashcode? => 4567fg^&R&HUIJ^&*(*9jiodfds)$%^&*BHNM




hash("20127538A") # return an integer




# 1377998101705154582%1000

# s[1377998101705154582] is empty or not?


# %%
