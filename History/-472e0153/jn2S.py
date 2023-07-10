# %%
import numpy as np; import pandas as pd; import matplotlib as mpl; import matplotlib.pyplot as plt; import seaborn as sns # data science lib
from collections import Counter
# from functools import reduce
# from itertools import combinations 
# import itertools
import math
from IPython.core.interactiveshell import InteractiveShell; InteractiveShell.ast_node_interactivity = "all"

# %%
class ListNode:
    def __init__(self, e, next=None):
        self.val=e
        self.next=next
    def __repr__(self):     
        if self:        
            return f"{self.val} -> {self.next}"
#------------ helper functions ----------------
def insert_all(nums):
    # return the head 
    cur = None
    for x in reversed(nums): # need to reverse nums first, since when inserting, the first val will become the last node
        new = ListNode(x, cur)
        cur = new
    return cur 

def print_nodes(head):
    while head:
        print(head.val, end=' ')
        head=head.next
    print()

#------------ helper functions ----------------


t=insert_all([1,2,3])
print(t)
# t=insert_all([1,2,3,4])
# middle_node(t).val
# %% [markdown] 
# linked list questions

# %%
# 287. Find the Duplicate Number: https://leetcode.com/problems/find-the-duplicate-number/
def find_duplicate(nums) -> int:
    n = len(nums) - 1
    total = (1+n)*(n/2)
    return sum(nums) - total

# nums = [1,3,4,2,2]
nums = [3,1,3,4,2]
find_duplicate(nums)
# %%
def reverse_list(head):
    # 206. Reverse Linked List: https://leetcode.com/problems/reverse-linked-list/

    # M1. iterative
    prev, cur = None, head
    # if we put next = cur.next here, we're screwed, since when cur reaches the last node, and if we do `next=next.next`, we have error
    while cur:
        next = cur.next # save the next node, so that we can still go to the next node after doing cur.next = prev
        cur.next = prev
        prev = cur
        cur = next
    return prev # since cur is already None

    # M2. recursion
    # idea: given head, reverse the whole list except the head, then head.next.next to head 
    if not head:
        return head # or None 

    new_head = head # for dealing with the case when the list has 1 node only...

    if head.next: # if the list has >=2 nodes...
        new_head = reverse_list(head.next)
        head.next.next = head # notice that we're doing head.next.next => our list must have >= 2 nodes!
        head.next = None

    return new_head

    # M3: another recursion. Just like M1 

    def helper(prev, cur):
        if cur is None:
            return prev 

        next = cur.next 
        cur.next = prev
        prev = cur
        cur = next 

        return helper(prev, cur)

    return helper(None, head)

t=insert_all([1,2,3])
reverse_list(t)
# print_nodes(reverse_list(t1))

# %%
def middle_node(head):
    # 876. Middle of the Linked List: https://leetcode.com/problems/middle-of-the-linked-list/
    # the difficulty stems from the fact we don't know the linked list size
    # If there are two middle nodes, return the second middle node

    # M0: use slow and fast pointer. Best
    # slow = fast = head
    # while fast and fast.next: # need two conditions. Consider list = [1,2]
    #     slow=slow.next
    #     fast=fast.next.next
    # return slow

    # M 0.5: deal with even case. Let slow point to a dummy node 
    # slow = ListNode(0, next = head)
    # fast = head
    # n = 0
    # while fast and fast.next: # need two conditions. Consider list = [1,2]
    #     slow=slow.next
    #     fast=fast.next.next
    #     n=+1
    # if n%2 != 0 
    #     return slow, slow.next
    # else:
    #     return slow.next

    # M1: put all nodes in a list first. Then return len()//2
    n, t, cur = 0, [], head
    while cur:
        t.append(cur)
        n+=1
        cur=cur.next
    return t[n//2]
    # return t[n//2-1],t[n//2] # handle n is even case

    # M2: traverse the whole linked list to get the size first. Then traverse again until reaching the middle node
    n, cur = 0, head
    while cur:
        n+=1
        cur=cur.next
    cur = head
    for i in range(n//2):
        cur = cur.next
    return cur

t=insert_all([1,2,3])
t=insert_all([1,2,3,4])
middle_node(t).val

# %%
def is_palindrom(head):
    # 234. Palindrome Linked List: https://leetcode.com/problems/palindrome-linked-list/

    # M1: create a list to store the val. Need extra space
    # t=[]
    # cur=head
    # while cur:
    #     t.append(cur.val)
    #     cur=cur.next
    # return t == t[::-1]

    # M2. create fast and slow pointers. Cons: need to modify our data structure
    slow=fast=head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    # slow is now in the middle, whereas fast is at the end 

    # reverse the second half
    prev = None
    while slow:
        next = slow.next
        slow.next = prev
        prev = slow
        slow = next
    
    # check if it is palindrome
    left, right = head, prev
    while right:
        if left.val != right.val:
            return False
        left = left.next
        right = right.next
    return True

    # another more advanced way
    # left, right = head, prev
    # while right and left.val == right.val:
    #     left = left.next
    #     right = left.next
        
    # return not right

t1 = insert_all([1,1,1,1])
# t1 = insert_all([1,1,1,1,1])
is_palindrom(t1)


# %%
def has_cycle(head):
    # 141. Linked List Cycle: https://leetcode.com/problems/linked-list-cycle/

    # M1: slow
    # cur, res = head, []
    # while cur:
    #     if cur in res:
    #         return True
    #     else: 
    #         res.append(cur)
    #         cur = cur.next
    # return False 

    # M2: based on M1, with faster lookup 
    cur, res = head, {}
    while cur:
        if cur in res:
            return True
        else: 
            res[cur] = 1
            cur = cur.next
    return False 

    # M3: Tortoise and Hare algo
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow is fast: return True
    return False


t1 = insert_all([1,2])
# t1.next.next = t1 
has_cycle(t1)


# %%
def get_intersection_node(head_A, head_B):
    # 160. Intersection of Two Linked Lists: https://leetcode.com/problems/intersection-of-two-linked-lists/

    # M1: put all nodes in list A in a set, then for list B, for each node, check whether it is in the set 
    # t = set()
    # cur = head_A
    # while cur:
    #     t.add(cur)
    #     cur = cur.next
    
    # cur = head_B
    # while cur:
    #     if cur in t:
    #         return cur
    #     cur = cur.next
    # return None

    # M2: make two pointers and let them start at the head of the two lists. When one reaches None, let it point to the head of another list and traverse again. The intersection point will be the result, or another possibility is each one is at the end of a list
    cur_A, cur_B = head_A, head_B
    while cur_A or cur_B:
        if not cur_A:
            cur_A = head_B
        if not cur_B:
            cur_B = head_A
        if cur_A is cur_B:
            return cur_A
        cur_A = cur_A.next
        cur_B = cur_B.next
    return None 
        
    # M2 # TODO
    # find the length first, and align their starting point. Complicated! 

# test 1
t1=insert_all([1,2])
t2=insert_all([3,4,5])
t3=insert_all([6,7])
t1.next.next = t3
t2.next.next.next = t3
print(get_intersection_node(t1, t2).val)

# test 2
t1=insert_all([1,2])
t2=insert_all([3,4,5])
print(get_intersection_node(t1, t2))

# test 3: it will detect the intersection without very soon, w/o each pointer going to another list
t1=insert_all([1,2])
t2=insert_all([3,4])
t3=insert_all([6,7])
t1.next.next = t3
t2.next.next = t3
print(get_intersection_node(t1, t2))
# %%
def remove_vals(head, val):
    # 203. Remove Linked List Elements (https://leetcode.com/problems/remove-linked-list-elements/)

    # M1: with dummy node
    dummy = ListNode(-1, head)
    prev, cur = dummy, head
    while cur:
        if cur.val == val:
            prev.next = cur.next
        else:
            prev = cur # or prev = prev.next
        cur = cur.next 
    return dummy.next

    # M2: bad. w/o dummy node
    # cur = cur_head = self.head
    # prev = None
    # # cur_head = self.head
    
    # while cur:
    #     if cur.val == val:
    #         if prev: # cur is not pointing to the first element 
    #             prev.next = cur.next
    #         else: # cur is still pointing to the first element 
    #             cur_head = cur.next # change head
    #     else:
    #         prev=cur
    #     cur=cur.next
    # self.head = cur_head


t=insert_all([1,2,3])
print(t)
print_nodes(remove_vals(t,1))
# %%
def delete_duplicates(head):
    # 83. Remove Duplicates from Sorted List: https://leetcode.com/problems/remove-duplicates-from-sorted-list/submissions/
    # very similar to removing vals, but in this problem, we can choose to remove the next value. But in the other problem, we must remove the cur val.

    # M1: two pointers. when cur is the same as previous, remove cur
    # dummy = ListNode(float('inf'), head)
    # prev, cur = dummy, head
    # while cur:
    #     if cur.val == prev.val:
    #         prev.next = cur.next
    #     else:
    #         prev = cur
    #     cur = cur.next
    # return dummy.next # or head 

    # M2: one pointer. look at the next node instead of previous, and so no need to have prev! If next is the same as cur, remove next
    cur = head
    while cur.next:
        next = cur.next
        if next.val == cur.val:
            cur.next = next.next
        else:
            cur = next
    return head

    # M2: bad. Need two pointers
    # if head:
    #     cur = head
    #     next = head.next 
    #     while next:
    #         if cur.val == next.val:
    #             cur.next = next.next 
    #         else:
    #             cur = next
    #         next = next.next
    # return head 

# t = insert_all([1,1,2,2,3,3,3])
t = insert_all([1,1,1])
print_nodes(delete_duplicates(t))
# %%
def delete_node(node):
    # 237. Delete Node in a Linked List: https://leetcode.com/problems/delete-node-in-a-linked-list/
    # Write a function to delete a node in a singly-linked list. You will not be given access to the head of the list, instead you will be given access to the node to be deleted directly.
    # It is guaranteed that the node to be deleted is not a tail node in the list.

    # M1: change its value to the next's node value. Then let its next point to the next next node. Therefore, this method doesn't work if the given node is the tail
    node.val = node.next.val
    node.next = node.next.next

    # M2: stupid method: copy next value to the current node, then loop.
    # next = node.next
    # while next:
    #     node.val = next.val
    #     if next.next:
    #         node = next 
    #     else:
    #         node.next = None
    #     next=next.next

t = insert_all([1,2,3])
delete_node(t.next)
print_nodes(t)

# %%
def merge_two_lists(l1,l2) -> ListNode:
    # 21. Merge Two Sorted Lists: https://leetcode.com/problems/merge-two-sorted-lists/
    # the two lists are sorted

    # M1: create a dummy node to avoid compicated edge cases. w/o dummy, we don't know which node to let cur point to at the beginning
    # dummy = ListNode(0, None)
    # cur = dummy
    # while l1 and l2:
    #     if l1.val < l2.val:
    #         cur.next = l1
    #         l1=l1.next
    #     else:
    #         cur.next = l2
    #         l2=l2.next
    #     cur = cur.next
    # cur.next = (l1 or l2) # let cur point to the list that is not yet exhausted
    # return dummy.next
        
    # M2: like M1, but use more complicated conditional to avoid the last step
    dummy = ListNode(0, None)
    cur = dummy
    while l1 or l2:
        if not l2 or (l1 and l1.val < l2.val):
            cur.next = l1
            l1=l1.next
        else:
            cur.next = l2
            l2=l2.next
        cur = cur.next
    return dummy.next


    # M3: complicated. does not create any new node
    # prev=None
    # head=None
    # while l1 or l2:
    #     if l2==None or (l1 and l2.val > l1.val):
    #         if prev:
    #             prev.next = l1
    #         else:
    #             head = l1
    #         prev = l1
    #         l1 = l1.next
    #     else:
    #         if prev:
    #             prev.next = l2
    #         else:
    #             head = l2
    #         prev = l2
    #         l2=l2.next
    # return head
    
    
    # M4: a solution that creates new list. Bad. Also, it's in reverse order
    # cur_l1 = l1
    # cur_l2 = l2
    # new_head = None
    # while cur_l1 or cur_l2:
    #     if cur_l2==None or (cur_l1 and cur_l2.val > cur_l1.val):
    #         new_node = ListNode(cur_l1.val, new_head)
    #         cur_l1=cur_l1.next
    #     else:
    #         new_node = ListNode(cur_l2.val, new_head)
    #         cur_l2=cur_l2.next
    #     new_head = new_node
    # return new_head
    
    # M5: recursion
    # this solution create a new linked list, without splicing the given nodes!!!
    # if l1 == None:
    #     return l2
    # elif l2 == None:
    #     return l1
    # if l1.val < l2.val:
    #     return ListNode(l1.val,merge_two_lists(l1.next,l2))
        
    # else:
    #     return ListNode(l2.val,merge_two_lists(l1,l2.next))
    
l1 = insert_all([1,2,4,5,6,7])
l2 = insert_all([1,3,4,10])
print_nodes(merge_two_lists(l1, l2))
# %%
def get_decimal_value(head):
    # 1290. Convert Binary Number in a Linked List to Integer: https://leetcode.com/problems/convert-binary-number-in-a-linked-list-to-integer/
    # - Given head which is a reference node to a singly-linked list. The value of each node in the linked list is either 0 or 1. The linked list holds the binary representation of a number. Return the decimal value of the number in the linked list.
    cur = head
    ans = 0
    while cur:
        ans = ans *2 + cur.val
        cur = cur.next 
    return ans 
    
# t = insert_all([0,1,1,0,1,0,1])
t = insert_all([1,1,1,1])
get_decimal_value(t)
# %%
def add_two_numbers(l1: ListNode, l2: ListNode) -> ListNode:
    # 2. Add Two Numbers: https://leetcode.com/problems/add-two-numbers/

    # idea: compute the digit and carry for each corresponding pair of nodes 
    # M1
    # dummy = ListNode(-1, None)
    # cur = dummy
    # carry = 0
    # while l1 or l2:
    #     # print('hi')
    #     first = l1.val if l1 else 0
    #     second = l2.val if l2 else 0

    #     # val = first+second + carry # can be two digits
    #     # carry = val // 10 
    #     # val = val % 10

    #     cur.next = ListNode((first+second)%10 + carry)
    #     cur = cur.next
    #     carry = (first+second)//10
    #     if l1:
    #         l1=l1.next
    #     if l2:
    #         l2=l2.next
    # if carry: cur.next = ListNode(carry)
    # return dummy.next

    # M1.5: add a condition for while loop so that it can deal with the final carry case too. Also, change the calculation of val and carry to avoid calculating first+second twice
    dummy = ListNode(-1, None)
    cur = dummy
    carry = 0
    while l1 or l2 or carry:
        first = l1.val if l1 else 0
        second = l2.val if l2 else 0
    
        val = first+second + carry # can be two digits
        carry = val // 10 
        val = val % 10
    
        cur.next = ListNode(val)
        cur = cur.next
        if l1:
            l1=l1.next
        if l2:
            l2=l2.next
    return dummy.next

t1 = insert_all([2,4,3])
t2 = insert_all([5,6,4])

t1 = insert_all([5,6,4])
t2 = insert_all([7,0,8])

t1 = insert_all([5,6,4,1])
t2 = insert_all([7,0,8])

t1 = insert_all([1])
t2 = insert_all([2])
print_nodes(add_two_numbers(t1,t2))
# %%
# 23. Merge k Sorted Lists: https://leetcode.com/problems/merge-k-sorted-lists/
# https://github.com/ChenglongChen/LeetCode-3/blob/master/Python/merge-k-sorted-lists.py

import heapq
def merge_k_lists(lists):
    h = []
    dummy = ListNode(0)
    cur = dummy

    for i, lis in enumerate(lists):
        if lis:
            print(lis.val)
            heapq.heappush(h, (lis.val, i, lis) )
    
    while h:
        val, i, node = heapq.heappop(h)
        cur.next = node
        cur = cur.next 
        if node.next:
            heapq.heappush(h, (node.next.val, i, node.next))
    return dummy.next

list1 = insert_all([1,4,5])
list2 = insert_all([1,3,4])
list3 = insert_all([2,6])
merge_k_lists([list1,list2,list3])
# %% [markdown] 
# backtracking questions
# %%
# https://www.youtube.com/watch?v=s9fokUqJ76A
# 22. Generate Parentheses: https://leetcode.com/problems/generate-parentheses/
def generate_parenthesis(n: int) -> list:
    # To understand
    stack = []
    res = []

    def backtrack(openN, closedN):
        if openN == closedN == n:
            res.append( "".join(stack) )
            # print(res)
            return
        
        if openN < n:
            stack.append("(")
            backtrack(openN+1, closedN)
            stack.pop()

            print(stack)
        
        if closedN < openN:
            stack.append(")")
            backtrack(openN, closedN+1)
            stack.pop()


    backtrack(0,0)
    return res
        
generate_parenthesis(3)
# print("hello")

# %%
def powerset(a):
    # if not a: return [[]]
    # temp=powerset(a[:-1])
    # return temp + [  subset + [a[-1]] for subset in temp]

    # backtracking
    res = []
    subset = []
    def dfs(i):
        if i>=len(a): # out of bound 
            res.append(subset.copy())
            return
    
        subset.append(a[i])
        dfs(i+1)

        print(subset)

        subset.pop()
        dfs(i+1)
    dfs(0)
    return res
    
a=[1,2,3]
print(powerset(a))
# %% [markdown] 
# array questions
# %%
def is_valid_sudoku(board: List[List[str]]) -> bool:
    # 36. Valid Sudoku: https://leetcode.com/problems/valid-sudoku/
    from collections import defaultdict
    rows = defaultdict(set)
    cols = defaultdict(set)
    squares = defaultdict(set)

    for i in range(9):
        for j in range(9):
            val = board[i][j]
            if val != '.' and (val in rows[i] or 
                                val in cols[j] or 
                                val in squares[(i//3, j//3)]):
                return False
            rows[i].add(val)
            cols[j].add(val)
            squares[(i//3, j//3)].add(val)
    return True

board = [["5","3",".",".","7",".",".",".","."]
,["6",".",".","1","9","5",".",".","."]
,[".","9","8",".",".",".",".","6","."]
,["8",".",".",".","6",".",".",".","3"]
,["4",".",".","8",".","3",".",".","1"]
,["7",".",".",".","2",".",".",".","6"]
,[".","6",".",".",".",".","2","8","."]
,[".",".",".","4","1","9",".",".","5"]
,[".",".",".",".","8",".",".","7","9"]]

# [["8","3",".",".","7",".",".",".","."]
# ,["6",".",".","1","9","5",".",".","."]
# ,[".","9","8",".",".",".",".","6","."]
# ,["8",".",".",".","6",".",".",".","3"]
# ,["4",".",".","8",".","3",".",".","1"]
# ,["7",".",".",".","2",".",".",".","6"]
# ,[".","6",".",".",".",".","2","8","."]
# ,[".",".",".","4","1","9",".",".","5"]
# ,[".",".",".",".","8",".",".","7","9"]]

is_valid_sudoku(board)
# %%
# 283. Move Zeroes: https://leetcode.com/problems/move-zeroes/
def move_zeros_end(a):
    # move all zeros to end, but keep the relative ordering
    l = r = 0
    # for i in range(len(a)):
    for n in a:
        if n !=0:
            a[l], a[r] = a[r], a[l]
            l+=1
        r+=1
    return a

a=[0,1,0,3,9]
move_zeros_end(a)
# %%
def longest_palindrome(s) -> str:
    # 5. Longest Palindromic Substring: https://leetcode.com/problems/longest-palindromic-substring/
    # brute force: TC => O(n^3)
    # n = len(s)
    # res = ''
    # n_res = 0 
    # for i in range(n):
    #     for j in range(i,n):
    #         candidate = s[i:j+1]
    #         if j-i+1>n_res and all(x==y for x,y in zip(candidate, reversed(candidate))):
    #             res=candidate
    #             n_res = j-i+1
    # return res

    # M2: for each char, start at it and then expand to the two directions
    n = len(s)
    res = ''
    n_res = 0 

    # len is odd case
    for i in range(n):
        l = r = i
        while l>=0 and r<=n-1 and s[l]==s[r]:
            if r-l+1 >n_res:
                res=s[l:r+1]
                n_res = r-l+1
            l-=1
            r+=1
    # return res

    # len is even case
    for i in range(n):
        l = i
        r = i + 1
        while l>=0 and r<=n-1 and s[l]==s[r]:
            if r-l+1 >n_res:
                res=s[l:r+1]
                n_res = r-l+1
            l-=1
            r+=1
    return res
    

s = "aaa"
longest_palindrome(s)
# %%
# hackerrank question
# https://www.hackerrank.com/challenges/encryption/problem
def encryption(s):
    s = s.replace(' ','') # remove space
    n = len(s) # string length after removing space
    print(n)
    # rows = int(n**0.5)
    n_columns = math.ceil(n**0.5)
    
    i = 0
    res=[]
    while i*n_columns <= n:
        res.append(s[i*n_columns:(i+1)*n_columns])
        i+=1
    

    output = []
    for i in range(n_columns):
        for row in res:
            try:
                output.append(row[i])
            except:
                pass
        output.append(' ')

    return ''.join(output[:-1])

s= 'hello world! I love you so much!!'
encryption(s)


# %%
# hackerrank question
"""
Given two strings, s and t, create a function that operates per the following rules:

Find whether string s is divisible by string t.  String s divisible by string t if string t can be concatenated some number of times to obtain the string s.
If s is divisible, find the smallest string, u, such that it can be concatenated some number of times to obtain both s and t.
If it is not divisible, set the return value to -1.
Return the length of the string u or -1.

Example 1

s = 'bcdbcdbcdbcd'

t = 'bcdbcd'


If string t is concatenated twice, the result is 'bcdbcdbcdbcd' which is equal to the string s.  The string s is divisible by string t. 

Since it passes the first test, look for the smallest string, u, that can be concatenated to create both strings s and t.

The string 'bcd' is the smallest string that can be concatenated to create both strings s and t. 

The length of the string u is 3, which is the integer value to return.


Example 2

s = "bcdbcdbcd"

t = "bcdbcd"


If string t is concatenated twice, the result is "bcdbcdbcdbcd" which is greater than string s.  There is an extra "bcd" in the concatenated string.

The string s is not divisible by string t, so return -1.
"""
def find_smallest_divisor(s, t):
    # Write your code here
    candidate = t
    is_divisor = False
    length_s = len(s)
    length_t = len(t)

    while len(candidate) <= length_s:
        if candidate == s:
            is_divisor = True
            break
        else:
            candidate += t

    if not is_divisor: return -1
        
    # part 2
    for i in range(1,len(t)+1):
        if length_t%i ==0 and (t[:i]*(length_t//i)) == t:
            return i


s = 'bcdbcdbcdbcd'
t = 'bcdbcd'

find_smallest_divisor(s, t)

# %%
# hackerrank question
def table_of_contents(text):
    # Write your code here
    cur_chapter = 0
    cur_section = 0
    res = []
    for sentence in text:
        if sentence[:2] == '# ':
            cur_section = 0 # new chapter. So need to restart the count
            cur_chapter +=1 
            res.append(f'{cur_chapter}. {sentence[2:]}')
        elif sentence[:3] == '## ':
            cur_section +=1
            res.append(f'{cur_chapter}.{cur_section}. {sentence[3:]}')
    return res 

text = ['# Algorithms',
'This chapter covers the most basic algorithms.',
'## Sorting',
'Quicksort is fast and widely used in practice',
'Merge sort is a deterministic algorithm',
'## Searching',
'DFS and BFS are widely used graph searching algorithms',
'Some variants of DFS are also used in game theory applications',
'# Data Structures',
'This chapter is all about data structures',
"It's a draft for now and will contain more sections in the future",
'# Binary Search Trees',]

table_of_contents(text)
# %%
# 121. Best Time to Buy and Sell Stock: https://leetcode.com/problems/best-time-to-buy-and-sell-stock/
def max_profit(prices: List[int]) -> int:
    # M1
    # buy = 0
    # sell = 0
    # profit = 0 
    # for i in range(1, len(prices)):
    #     if prices[i] < prices[buy]:
    #         buy = i
    #         sell = i
    #     elif prices[i] > prices[sell]:
    #         sell = i
    #         profit = max(prices[sell] - prices[buy], profit)
    # # print(buy, sell)
    # return profit

    # M2
    buy, sell = 0,1
    profit = 0
    while sell < len(prices):
        if prices[buy] < prices[sell]:
            profit = max(profit, prices[sell] - prices[buy])
        else:
            buy = sell
        sell +=1
    return profit
# prices = [7,1,5,3,6,4]
prices = [3,2,6,5,0,3]
max_profit(prices)
# %%
# 122. Best Time to Buy and Sell Stock II: https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/
def max_profit(prices: List[int]) -> int:
    # at any time, we can hold one stock only!
    profit = 0
    for i in range(1, len(prices)):
        if prices[i] > prices[i-1]:
            profit += prices[i] - prices[i-1]
    return profit

prices = [7,1,5,3,6,4]
prices = [1,2,3,4,5]
prices = [7,6,4,3,1]
max_profit(prices)
# %%
def is_valid_subsequence(array, sequence):
    # Write your code here.
    # i = 0
    # for x in array:
    # 	if x == sequence[i]:
    # 		i+=1
    # 		if i == len(sequence): return True
    # return False

    # M2
    arr_idx = seq_idx = 0
    while arr_idx < len(array) and seq_idx < len(sequence):
        if array[arr_idx] == sequence[seq_idx]:
            seq_idx+=1
        arr_idx+=1
    return seq_idx == len(sequence)

array=[5,1,22,25,6,-1,8,10]
sequence=[1,6,-1,10]
is_valid_subsequence(array, sequence)
# %%
# https://www.hackerrank.com/challenges/ctci-array-left-rotation/problem?h_l=interview&playlist_slugs%5B%5D=interview-preparation-kit&playlist_slugs%5B%5D=arrays
def rotate_left(a, d):
    # Write your code here

    # M1: naive
    n = len(a)
    d = d%n
    for i in range(d):
        temp = a[0]
        for i in range(n-1):
            a[i] = a[i+1]
        a[-1] = temp
    return a

    # M2: for a[i], it should be placed in a[(i-d)%n]
    def rotate_left(a, d):
        n = len(a)
        d = d%n
        temp = a.copy()
        for i in range(len(a)):
            a[(i-d)%n] = temp[i]
        return a

a, d =[1,2,3], 5
rotate_left(a,d)
    
    # M3: use slicing in python 
    # return a[d:] + a[:d]


a, d =[1,2,3], 4
rotate_left(a,d)

# %%
# https://www.hackerrank.com/challenges/2d-array/problem?h_l=interview&playlist_slugs%5B%5D=interview-preparation-kit&playlist_slugs%5B%5D=arrays
def cal_sum(arr, i, j):
    sum =0
    for row in range(i,i+3):
        for col in range(j,j+3):
            if not (row == i+1 and (col == j or col == j+2)):
                sum += arr[row][col]
    return sum

def hourglass_sum(arr):
    row = len(arr)
    col = len(arr[0])
    res=[]
    for i in range(row - 2):
        for j in range(col - 2):
            res.append(cal_sum(arr, i, j))
    return res

arr=[[-9, -9, -9,  1, 1, 1 ],
[ 0, -9,  0,  4, 3, 2],
[-9, -9, -9,  1, 2, 3],
[ 0,  0,  8,  6, 6, 0],
[ 0,  0,  0, -2, 0, 0],
[ 0,  0,  1,  2, 4, 0]]

hourglass_sum(arr)
# %%
# https://www.hackerrank.com/challenges/new-year-chaos/problem?h_l=interview&playlist_slugs%5B%5D=interview-preparation-kit&playlist_slugs%5B%5D=arrays
def minimum_bribes(q):
    # TODO: this is a wrong code
    total = 0
    for i, num in enumerate(q):
        if num > i+1: # has bribe
            bribe = num-(i+1)
            if bribe >2:  
                print("Too chaotic")
                return
            total += num-(i+1)
    print(total)

# q=[1,2,3,5,4,6,7,8]
# q=[2,1,5,3,4]
q= [1, 2, 5, 3, 7, 8, 6, 4,]

minimum_bribes(q)
    

# %%
def three_sum_leetcode(nums):
    # TODO
    # Naive 
    n = len(nums)
    res=[]
    for i in range(n-1):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if nums[i]+nums[j]+nums[k]==0:
                    # res.append([i,j,k])
                    res.append([nums[i],nums[j],nums[k]])
    return res
nums = [-1,0,1,2,-1,-4]
three_sum_leetcode(nums)
# %%
def checkStraightLine(coordinates: List[List[int]]) -> bool:
    # TODO
    x1, y1 = coordinates[0]
    x2, y2 = coordinates[1]
    for x, y in coordinates[2:]:
        if (y2 - y1) * (x - x1) != (x2 - x1) * (y - y1):
            return False
    return True
# %%
def two_sum(nums, target):
    # 1. Two Sum: https://leetcode.com/problems/two-sum/
    # Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.You may assume that each input would have exactly one solution, and you may not use the same element twice.
    
    # M1: pre-build dict with (key, index) pairs first. Not optimal 
    # d = {num:i for i,num in enumerate(nums)} # or, d = dict( zip(nums, range(len(nums)))  )
    # for i, num in enumerate(nums):
    #     diff = target - num
    #     idx = d.get(diff, -1)
    #     if idx not in [-1, i]: # exclude i because we can't use the same number again 
    #         return i, idx
    # return None

    # M2. build dict incrementally. Works even when there are duplicates. Best
    d={}
    for i, num in enumerate(nums):
        diff = target - num
        idx = d.get(diff, None)
        if idx is not None: 
            return i, idx
        d[num] = i 
    return None

    # Naive
    # n = len(nums)
    # for i in range(n-1):
    #     for j in range(i+1, n):
    #         if nums[i]+nums[j] == target:
    #             return i,j

nums = [3,2,4]
target = 6
# nums = [2,7,11,15,7,7,7,1]
# target = 9
two_sum(nums, target)
# %%
# 88. Merge Sorted Array: https://leetcode.com/problems/merge-sorted-array/ (!)
# You are given two integer arrays nums1 and nums2, sorted in non-decreasing order, and two integers m and n, representing the number of elements in nums1 and nums2 respectively.
# Merge nums1 and nums2 into a single array sorted in non-decreasing order.
# The final sorted array should not be returned by the function, but instead be stored inside the array nums1. To accommodate this, nums1 has a length of m + n, where the first m elements denote the elements that should be merged, and the last n elements are set to 0 and should be ignored. nums2 has a length of n.

def merge(nums1: list, m: int, nums2: list, n: int) -> None:
    # M1: maintain three pointers. one starts from nums1[-1], another starts from the nums1[m-1], and last one start from the end of nums2[n-1]. 
    i = m-1
    j = n-1 
    for k in range(m+n-1, -1, -1):
        if j<0 or (i>=0 and nums1[i]> nums2[j]):
            nums1[k] = nums1[i]
            i-=1
        else:
            nums1[k] = nums2[j]
            j-=1
    return nums1
nums1 , m, nums2, n = [1,2,3,0,0,0], 3,  [2,5,6], 3
nums1 , m, nums2, n = [1,5,9,0,0,0,0,0,0], 3,  [-1,-1,2,3,4,10], 6

merge(nums1,m, nums2, n)
# %%
# %% [markdown] 
# heaps
# %%
import heapq
def k_smallest_pairs(nums1: List[int], nums2: List[int], k: int) -> List[List[int]]:
    # 373. Find K Pairs with Smallest Sums: https://leetcode.com/problems/find-k-pairs-with-smallest-sums/
    # similar to 599 minimum index sum 


# %%
import heapq
def k_closest(points: List[List[int]], k: int) -> List[List[int]]:
    # 973. K Closest Points to Origin: https://leetcode.com/problems/k-closest-points-to-origin/

    # M1: use heappush
    # h = []
    # for x,y in points:
    #     # print(point)
    #     dist = x**2 + y**2
    #     heapq.heappush(h, (dist,[x,y])) 
    # return [heapq.heappop(h)[1] for i in range(k) ]
    
    # M1.5: slight optimization
    # h = []
    # for i, x,y in enumerate(points):
    #     # print(point)
    #     dist = x**2 + y**2
    #     heapq.heappush(h, (dist,i)) 
    # return [points[heapq.heappop(h)[1]] for i in range(k) ]

    # M2: use heapify 
    h = [0]*len(points)
    for i, (x, y) in enumerate(points):
        dist = x**2 + y**2
        h[i] = [dist,x,y] # get x,y, since at the end we need to return the points
    
    heapq.heapify(h)

    return [heapq.heappop(h)[1:] for i in range(k)]

points, k = [[1,3],[-2,2]], 1
points, k = [[3,3],[5,-1],[-2,4]],2

k_closest(points, k)
# %%
# 215. Kth Largest Element in an Array: https://leetcode.com/problems/kth-largest-element-in-an-array/
import heapq
def find_kth_largest(nums: List[int], k: int) -> int:
    n = len(nums)
    heapq.heapify(nums)
    for i in range(n-k):
        heapq.heappop(nums)
    return heapq.heappop(nums)
    
# nums,k = [3,2,1,5,6,4], 2
nums, k = [3,2,3,1,2,4,5,5,6],4
find_kth_largest(nums, k)
# %%
import heapq
from collections import Counter
def top_k_frequent(nums: List[int], k: int) -> List[int]:
    # 347. Top K Frequent Elements: https://leetcode.com/problems/top-k-frequent-elements/
    # Given an integer array nums and an integer k, return the k most frequent elements. You may return the answer in any order.

    # M0: sorting. Slow
    counts = Counter(nums)
    h = list(zip( counts.values(), counts.keys()))
    sorted(h)
    return [h[i][1] for i in range(k)]

    # M1: heap
    counts = Counter(nums)
    h = list(zip( counts.values(), counts.keys())) # or h = [[freq, num] for num, freq in counts.items()]
    print(h)
    
    heapq.heapify(h)
    n=len(h)
    for i in range(n-k): # remove the top n-k pairs with lowest frequencies 
        heapq.heappop(h)
    return [pair[1] for pair in h]

    # M2: clever tricks: use bucket-sort like method to build a (freq:val) pairs and use dict 
    counts = Counter(nums)
    inverse={}

    for val,freq in counts.items():
        inverse.setdefault(freq,[]).append(val)
    # print(inverse)

    res=[]

    for freq in sorted(inverse.keys(), reverse=True): # slow, because it requires sorting..
        for x in inverse[freq]:
            res.append(x)
            k-=1
            if k==0:
                return res


    # M3: clever tricks: use bucket-sort like method and use list of lists
    counts = Counter(nums)
    n = len(nums)
    
    freq_to_val = [ [] for i in range(n+1)]
    for val, freq in counts.items():
        freq_to_val[freq].append(val)

    res=[]
    for i in range(n, 0, -1): # to 1 only, since no elements with freq = 0 !
        for x in freq_to_val[i]:
            res.append(x)
            k-=1
            if k==0:
                return res

# nums, k = [1,1,1,2,2,3], 2
nums, k = [1,1,1,2,2,2,3], 2
# nums, k = [1], 1
top_k_frequent(nums, k)
# %%

# %% [markdown] 
# binary tree
# %%
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def print_by_level(root):
    q = deque([root])
    while q:
        for _ in range(len(q)):
            node = q.popleft()
            print(node.data, end=' ')
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
        print(' ')
# %%
# 98. Validate Binary Search Tree: https://leetcode.com/problems/validate-binary-search-tree/
def is_valid_BST(node):
    # TODO: Wrong sol
    if root is None: return True

    if root.left:
        if root.left.val > root.val:
            return False

    if root.right:
        if root.right.val > root.val:
            return False
    
    return is_valid_BST(node.left) and is_valid_BST(node.right)


    # cur = node
    # while cur:
    #     val = cur.val
    #     left = cur.left
    #     right = cur.right
    #     while left:
    #         if left.val >= val:
    #             return False
    #     while right:
    #         if right.val >= val:
    #             return False
    # return True

# %%
# 108. Convert Sorted Array to Binary Search Tree: https://leetcode.com/problems/convert-sorted-array-to-binary-search-tree/ (!)
def sorted_array_to_bst(nums: List[int]) -> TreeNode:
    def helper(l, r):
        if l>r: return None
        mid = (l+r)//2
        root = Node(nums[mid])
        root.left = helper(l, mid-1)
        root.right = helper(mid+1, r)
        return root 
    return helper(0, len(nums)-1)

nums = [-10,-3,0,5,9]
print_by_level(sorted_array_to_bst(nums))
# %%
# 230. Kth Smallest Element in a BST: https://leetcode.com/problems/kth-smallest-element-in-a-bst/
def kth_smallest(root: TreeNode, k: int) -> int:
    # naive. Use dfs
    res = []
    def helper(node):
        if node is None: return None
        helper(node.left)
        res.append(node)
        helper(node.right)
    helper(root)
    return res[k-1]

    # NOT WORK:
    # def helper(node, count):
    #     if node is None: return None
    #     helper(node.left, count)
    #     count +=1
    #     if count == k:
    #         return node.val
    #     helper(node.right, count)
    # return helper(root, 0)

    # M2: iterative solution
    n = 0 # numbe
    stack = []
    cur = root 
    while cur or stack:
        while cur:
            stack.append(cur)
            cur = cur.left
        
        cur = stack.pop()
        n+=1
        if n==k:
            return cur.val
        cur = cur.right

root = TreeNode(12)
root.left = TreeNode(13)

print(kth_smallest(root, 1))
# %%

# %% [markdown] 
# dynamic programming
# %%
# 70. Climbing Stairs:https://leetcode.com/problems/climbing-stairs/
def climb_stairs(n: int) -> int:
    res = {1:1, 2:2}
    for i in range(3, n+1):
        res[i] = res[i-1] + res[i-2]
    return res[n]

climb_stairs(3)
# %%
# 198. House Robber: https://leetcode.com/problems/house-robber/
def rob(nums: list) -> int:
    # M1. Slow; requires comouting len(nums) and extra storage
    # n = len(nums)
    # if n == 1: return nums[0] # n>=1
    # res = {1:nums[0], 2: max(nums[0], nums[1])}
    # for i in range(3, n+1):
    #     res[i] = max(nums[i-1] + res[i-2] , res[i-1])
    # return res[n]

    # M2
    rob1 = rob2 = 0
    for n in nums:
        temp = max(n + rob1, rob2)
        rob1 = rob2
        rob2 = temp
    return temp 


nums = [2,7,9,3,1]
rob(nums)
# %%
# 300. Longest Increasing Subsequence:https://leetcode.com/problems/longest-increasing-subsequence/ (!)
"""
- very good question!!
- idea: start from the end, not from the beginning!
- LIS[i] = LIS starting at index i. 
- ans = max(LIS)
- to find LIS[i], look at LIS[i+1], LIS[i+2]...,so need to start from the end!
- if nums[i] < nums[j], that means we can consider picking LIS[j] + 1, since our sequence starts at i, where i < j
"""

def length_of_LIS(nums):
    n = len(nums)
    LIS = [1]*n # LIS[i] = LIS starting at index i. ans = max(LIS)
    for i in range(n-1, -1, -1):
        for j in range(i+1, n):
            if nums[j] > nums[i]:
                LIS[i] = max(LIS[i], 1 + LIS[j])
    return LIS

nums = [10,9,2,5,3,7,101,18]
length_of_LIS(nums)

# %%
# 1911. Maximum Alternating Subsequence Sum: https://leetcode.com/problems/maximum-alternating-subsequence-sum/
def max_alternating_sum(nums: list) -> int:
    cur = 0
    for i, n in enumerate(nums):
        temp = max((-1)**i * nums[i] + cur, cur) 
        cur = temp
    return cur 
nums = [6,2,1,2,4,5]
max_alternating_sum(nums)
# %%
# 56. Merge Intervals: https://leetcode.com/problems/merge-intervals/
def merge(intervals: List[List[int]]) -> List[List[int]]:
    intervals.sort(key = lambda i: i[0])
    ans = [intervals[0]]

    for start, end in intervals[1:]:
        last_end = ans[-1][1]
        if start <= last_end:
            ans[-1][1] = max(end, last_end)
        else:
            ans.append([start, end])

    return ans

# intervals = [[1,3],[8,10],[2,6],[15,18]]
intervals = [[1,4],[4,5]]
merge(intervals)

