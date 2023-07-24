def merge_dicts(dict1, dict2):
    # Check for overlapping keys
    intersection = set(dict1.keys()) & set(dict2.keys())
    if intersection:
        raise ValueError(f"Overlapping keys found: {intersection}")

    # Merge the dictionaries
    merged_dict = dict1.copy()
    merged_dict.update(dict2)
    
    return merged_dict

# Example usage:
dict1 = {'name': 'John', 'age': 30}
dict2 = {'city': 'New York', 'age': 35}

try:
    merged_dict = merge_dicts(dict1, dict2)
    print(merged_dict)
except ValueError as e:
    print(f"Error: {e}")