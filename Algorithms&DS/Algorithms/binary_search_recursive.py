def recursive_binary_search(list, target):
    """ We are not returning the index of the list like sorted_binary_search, because we are change the list constantly"""
    if len(list) == 0:
        return False
    else:
        midpoint = len(list) //2
    
    if list[midpoint] == target:
        return midpoint
    else:
        if list[midpoint] < target:
            return recursive_binary_search(list[midpoint+1:], target)
        else:
            return recursive_binary_search(list[:midpoint], target)


a = [1 ,2 ,3, 4, 5, 6, 7, 8, 9, 10]

ans = recursive_binary_search(a, 4)
print(ans)