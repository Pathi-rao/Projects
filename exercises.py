"""
Given a one-dimensional array, write a function that returns the length of the longest sequence of 1s. The array consists of integers, and it contains only 1s and 0s.
"""
a = [0, 0, 0, 0, 1, 1]

counter = 0

for i in range(len(a)):
    # print(a[i], a[i+1])

    if i+1 == len(a):
        # print(counter)
        break
        
    if a[i] and a[i+1] == 1:
        counter += 1
print(counter)

# TODO: Include last and first element of the list as a pair too
