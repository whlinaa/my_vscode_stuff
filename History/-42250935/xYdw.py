# %%

for i in range(1, 8):
    print(i)
    n = 10**7

    t = list(range(n))
    s = set(range(n))

    # print(t)
    # print(s)

    %time print(-10 in t)
    %time print(-10 in s)