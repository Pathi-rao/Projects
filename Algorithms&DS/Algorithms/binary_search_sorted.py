def sorted_binary_search(list, target):
    """ When the given input list is already sorted """
    first = 0
    last = len(list) - 1
    while first <= last:
        midpoint = (first + last) // 2
        if list[midpoint] == target:
            return midpoint
        elif list[midpoint] <= target:
            first = midpoint + 1
        else:
            last = midpoint - 1
    return None


a = [1 ,2 ,3, 4, 5, 6, 7, 8, 9, 10]

ans = sorted_binary_search(a, 2)
print(ans)