# %%
n = 10**1

t = list(range(n))
s = set(range(n))

print(t)
print(s)

%time print(-10 in t)
%time print(-10 in s)