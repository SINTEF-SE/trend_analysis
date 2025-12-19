import random
from pydantic import BaseModel, Field, create_model
from typing import Literal

import yaml
with open('taxonomies/ai_taxonomies.yaml', 'r') as file:
    AI_TAXONOMY= yaml.safe_load(file)
AI_TAXONOMY = AI_TAXONOMY["AI Taxonomy"]


def find_child_nodes(tree, path):
    subtree = tree
    for key in path:
        if type(subtree) is list: # if subtree is a list, and there is still a key in path, the key is a list element, i.e. leaf node (there are no child nodes)
            return []
        try:
            subtree = subtree[key]
        except:
            print(list(subtree.keys()), key)
            raise KeyError
    if type(subtree) is dict:
        return list(subtree.keys())
    elif subtree is None: # no child nodes
        return []
    else:
        assert type(subtree) is list, (subtree, path, type(subtree))
        return subtree


class Traverser:
    def __init__(self, tree, start_path=[], shuffle_alternatives = True):
        self.TREE = tree
        self.start_path = start_path
        self.reset_position(start_path)
        self.shuffle = shuffle_alternatives

    def set_traversal_type(self,traversal_type):
        if traversal_type == "down":
            self.include_parents = False
            self.include_siblings = False
        elif traversal_type == "vertical":
            self.include_parents = True
            self.include_siblings = False
        elif traversal_type == "free":
            self.include_parents = True
            self.include_siblings = True
        else:
            raise ValueError


    def get_child_nodes(self):
        if self.current_path[-1] == "Other":
            raise ValueError
        result = find_child_nodes(self.TREE, self.current_path).copy()
        if self.shuffle:
            random.shuffle(result)
        if not "Not relevant" in result and len(result): # only add other option if there is no "not relevant" and its not already a leaf node
            result.append("Other")
        return result

    def get_sibling_nodes(self):
        siblings = find_child_nodes(self.TREE, self.current_path[:-1]).copy()
        siblings.remove(self.current_path[-1])
        return siblings
    def get_parent_nodes(self):
        # NOTE: for now, we assume single parent
        if len(self.current_path) < 2:
            return []
        return [self.current_path[-2]]

    def move(self, new_node):
        if new_node in self.get_child_nodes():
            self.current_path.append(new_node)
            return "v"
        elif new_node in self.get_parent_nodes() and self.include_parents:
            self.current_path = self.current_path[:-1]
            return "^"
        elif new_node in self.get_sibling_nodes() and self.include_siblings:
            self.current_path = self.current_path[:-1] + [new_node]
            return "<"
        else:
            raise ValueError # New node is not in the alternatives


    def get_pydantic_form(self):
        possible_values= self.get_child_nodes() + [self.current_path[-1]]
        if self.include_parents:
            possible_values.extend(self.get_parent_nodes())
        if self.include_siblings:
            possible_values.extend(self.get_sibling_nodes())
        assert len(possible_values) == len(set(possible_values)), possible_values # Checks that there is no duplicate. otherwise llm output is ambiguous
        if len(possible_values) == 1: # no reason to keep generating if there is only one possibility
            assert possible_values[0] == self.current_path[-1]
            raise StopIteration
        if self.shuffle:
            random.shuffle(possible_values)

        field_type = Literal[tuple(possible_values)]# make Literal dynamically by converting to tuple
    
        fieldname = "field_"+self.current_path[-1].replace(" ","_").replace("-","_").replace(",", "_").replace("/","_")
        field = Field(description = "Most relevant subnode")
        pydantic_form = create_model(fieldname, **{"next_part_of_path":(field_type, field)})
        return pydantic_form
    
    def get_field(self):
        pydantic_form = self.get_pydantic_form()
        return next(iter(pydantic_form.model_fields.values()))

    def reset_position(self, path=None):
        if path is None:
            self.current_path = self.start_path.copy()
        else:
            self.current_path = path.copy()


class v4_Schema(BaseModel):
    arch : list[str] = Field(description = "Model architecture")
    prob : list[str] = Field(description = "AI problem type")
    para : list[str] = Field(description = "Learning paradigm")
    appl : list[str] = Field(description = "Application domain")

def get_v4_traverser_dict():
    traversers = {
            "arch": Traverser(AI_TAXONOMY, ["Model architecture"]),
            "prob": Traverser(AI_TAXONOMY, ["AI problem type"]),
            "para": Traverser(AI_TAXONOMY, ["Learning paradigm"]),
            "appl": Traverser(AI_TAXONOMY, ["Application domain"]),
            }
    return traversers



if __name__ == "__main__":

    t = Traverser(AI_TAXONOMY, ["Model architecture"])
    t.set_traversal_type("down")

    t.reset_position(["Application domain"])
    t.move("General-purpose applications")
    t.move("Assistants and chat")
    t.move("Chatbots")
    assert "Open-domain/social chatbots" in t.get_child_nodes()
    try:
        t.move("Customer support assistants")
        raise AssertionError
    except ValueError:
        pass
    t.set_traversal_type("free")
    t.move("Customer support assistants")
    assert len(t.get_child_nodes()) == 0
    print(0)
