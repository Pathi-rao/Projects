def linear_search(list, target):
    for i in range(0, len(list)):
        if list[i] == target:
            return i
    return None


index = linear_search([1 ,2 ,3, 4, 5], 10)
print(index)