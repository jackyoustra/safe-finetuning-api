from typing import Dict, TypeVar, List, Union, overload

T = TypeVar('T')
K = TypeVar('K', bound=Union[int, str])

@overload
def sort_dict_by_int_key(input_dict: Dict[int, T]) -> List[T]: ...

@overload
def sort_dict_by_int_key(input_dict: Dict[str, T]) -> List[T]: ...

def sort_dict_by_int_key(input_dict: Dict[K, T]) -> List[T]:
    try:
        # Convert keys to integers if they're strings, or use as-is if they're already integers
        int_key_items = []
        for key, value in input_dict.items():
            if isinstance(key, str):
                int_key_items.append((int(key), value))
            elif isinstance(key, int):
                int_key_items.append((key, value))
            else:
                raise ValueError(f"Key '{key}' is neither a string nor an integer")
        
        # Sort the list based on the integer keys
        sorted_items = sorted(int_key_items, key=lambda x: x[0])
        
        # Return only the values in the sorted order
        return [value for _, value in sorted_items]
    except ValueError as e:
        raise ValueError("All keys must be integers or strings convertible to integers") from e

# Example usage:
if __name__ == "__main__":
    # Example 1: Valid input with string keys
    valid_dict_str: Dict[str, str] = {"1": "apple", "3": "banana", "2": "cherry"}
    try:
        result_str = sort_dict_by_int_key(valid_dict_str)
        print("Sorted string-key values:", result_str)
    except ValueError as e:
        print("Error:", str(e))

    # Example 2: Valid input with integer keys
    valid_dict_int: Dict[int, int] = {1: 10, 3: 30, 2: 20}
    try:
        result_int = sort_dict_by_int_key(valid_dict_int)
        print("Sorted integer-key values:", result_int)
    except ValueError as e:
        print("Error:", str(e))

    # Example 3: Valid input with mixed keys
    valid_dict_mixed: Dict[Union[int, str], str] = {1: "one", "2": "two", 3: "three", "4": "four"}
    try:
        result_mixed = sort_dict_by_int_key(valid_dict_mixed)
        print("Sorted mixed-key values:", result_mixed)
    except ValueError as e:
        print("Error:", str(e))

    # Example 4: Invalid input (non-integer string key)
    invalid_dict: Dict[str, str] = {"1": "apple", "two": "banana", "3": "cherry"}
    try:
        result_invalid = sort_dict_by_int_key(invalid_dict)
        print("Sorted values:", result_invalid)
    except ValueError as e:
        print("Error:", str(e))
