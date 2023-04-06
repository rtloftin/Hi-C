"""Implements grid-search over training configurations"""
from collections import namedtuple
from copy import deepcopy

Parameter = namedtuple("Parameter", ["key", "value"])

def get_parameters(dictionary, base_key=[]):
    """Recursively searches through a dictionary and returns all tunable parameters"""

    # Convert arrays to dictionaries if needed
    if isinstance(dictionary, list):
        dictionary = {idx:value for idx, value in enumerate(dictionary)}

    if "grid_search" in dictionary:
        assert len(dictionary) == 1, "'grid_search' entry must be the unique child of its parent"
        assert isinstance(dictionary["grid_search"], list), "'grid_search' value must be a list of parameters"

        return [Parameter(base_key, dictionary["grid_search"])]
    else:
        parameters = []
        for key, value in dictionary.items():
            if isinstance(value, (dict, list)):  # NOTE: Recursive processing will stop for arrays - will break tuning
                parameters += get_parameters(value, base_key + [key])
        
        return parameters


def serialize_dict(dictionary):
    """converts a nested dict to a string"""

    items = []
    for key, value in dictionary.items():
        if isinstance(value, dict):
            items.append(f"{key}={serialize_dict(value)}")
        elif isinstance(value, list):
            items.append(f"{key}={serialize_list(value)}")
        else:
            items.append(f"{key}={value}")
    
    return f"({'_'.join(items)})"


def serialize_list(l):
    """converts a nested list to a string"""

    items = []
    for value in l:
        if isinstance(value, dict):
            items.append(serialize_dict(value))
        elif isinstance(value, list):
            items.append(serialize_list(value))
        else:
            items.append(value)
    
    return f"({'_'.join(items)})"


def set_recursive(dictionary, key, value):
    """Sets a value in a nested dictionary"""

    for idx in range(len(key) - 1):
        dictionary = dictionary[key[idx]]

    dictionary[key[-1]] = value


def get_variations(base_name, base_config, free_parameters, set_parameters=[]):
    """Takes a list of tunable parameters and generates a list of configurations"""

    if len(free_parameters) == 0:
        name = []

        for param in set_parameters:
            value = param.value

            if isinstance(value, dict):
                value = serialize_dict(value)
            elif isinstance(value, list):
                value = serialize_list(value)
            
            name.append(f"{param.key[-1]}={value}")

        name = '_'.join(name)
        name = f"{base_name}_{name}"

        config = deepcopy(base_config)

        for p in set_parameters:
            set_recursive(config, p.key, p.value)

        return {name: config}
    else:
        variations = {}

        for value in free_parameters[0].value:
            parameter = Parameter(free_parameters[0].key, value)
            variations.update(get_variations(base_name, 
                                             base_config, 
                                             free_parameters=free_parameters[1:],
                                             set_parameters=set_parameters + [parameter]))

        return variations

                                     
def grid_search(name, config):
    parameters = get_parameters(config)

    if len(parameters) == 0:
        return None
    else:
        parameters.sort(key=lambda param: param.key[-1])
        return get_variations(name, config, parameters)
