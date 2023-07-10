# %%

for i in range(5, 8):
    print(i)
    n = 10**i

    t = list(range(n))
    s = set(range(n))

    # print(t)
    # print(s)

    %timeit print(-10 in t)
    %timeit print(-10 in s)