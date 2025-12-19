import outlines
from typing import Literal

def make_regex_string(field_type, min_l, max_l, answer_in_quotes):
    # get regex string depending on type
    allowed_string_chars = r'''[a-zA-Z 0-9'-,\(\)]''' # any sequence of words, numbers, spaces, and any of '-,()
    if not answer_in_quotes:
        allowed_string_chars = allowed_string_chars.replace("'", "") # "'" would cause trouble 
    regex_dict = {
            int : outlines.types.integer.pattern,
            str : allowed_string_chars + "*", # any length
            float : outlines.types.number.pattern,
            bool : outlines.types.boolean.pattern,
            }
    if field_type in regex_dict:
        regex = regex_dict[field_type]
    elif getattr(field_type, "__origin__", None) is Literal:
        choices = field_type.__args__
        regex = r"(" + r"|".join(choices) + r")"
    else:
        raise NotImplementedError

    # get constrained regex string if relevant
    if not (min_l is None and max_l is None):
        if field_type == str: # constr

            if min_l is None:
                min_l = 4 # TODO make this more clean? It should of course be 0, but it is better to predict something in our case (should be included in the schema ideally)
            if max_l is None:
                max_l = ""
            regex = allowed_string_chars + "{" + str(min_l) + "," + str(max_l) + "}"
        else:
            raise NotImplementedError

    # add quotes
    if answer_in_quotes:
        regex = '"'+regex+'"'


    # print regex, to verify/look for bugs
    #print(f"Made regex for {field_type, min_l, max_l}: {regex}")
    return regex

def make_constrained_generator(field_type, llm_model, min_l, max_l, answer_in_quotes, sampler = None):
    """
    make an outlines generator restricted to a specific type*, potentially with constraints, using regex to describe output restrictions 
    * the output is always a string, but a parsable one
    """

    # get regex
    regex = make_regex_string(field_type, min_l, max_l, answer_in_quotes)

    # make outlines generators
    generator = outlines.generate.regex(llm_model, regex, sampler=sampler)
    return generator
