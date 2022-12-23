"""
To move in a dir dedicated to tests

"""
# %%


arg1 = 0.2
arg2 = 0.3
arg3 = 0.4
title_reprint = 10

arg_dict = {
    "arg1": arg1,
    "arg2": arg2,
    "arg3": arg3,
}

def print_vars_table(arg_dict: dict, print_title: bool=False) -> None:
    if print_title:
        title = "|" + "|".join(
            [f"{arg_name:^10}" for arg_name in arg_dict.keys()]) + "|"
        title += "\n|" + "|".join(
            [f"{'----------':^10}" for arg_name in arg_dict.keys()]) + "|"
        print(title)

    values = "|" + "|".join(
            [f"{arg_value:^10}" for arg_value in arg_dict.values()]) + "|"
    print(values)
    return None

print_vars_table(arg_dict)

# %%