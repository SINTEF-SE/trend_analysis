
class Node:
    def __init__(self, parent, label):
        self.parent = parent
        self.label = label
        self.children = []
        #self.depth = 0 if parent is None else parent.depth + 1

    def set_child(self, child):
        child.parent = self
        #child.depth = self.depth + 1
        self.children.append(child)

    def set_depth(self, depth):
        self.depth = depth
        for child in self.children:
            child.set_depth(depth + 1)

    def get_max_depth(self):    
        self.max_depth = max([self.depth, *[child.get_max_depth() for child in self.children]])
        return self.max_depth

    def set_dfs(self, dfs):
        self.dfs = {}
        for key in dfs:
            df = dfs[key]
            if self.parent and self.parent.parent:
                self.dfs[key] = df[df["predicted_tag"].str.contains("'"+self.parent.label+"', '"+self.label+"'")] # Make sure its the ones where this branch is not relevant
            else: # main root node and root for each taxonomy
                self.dfs[key] = df[df["predicted_tag"].str.contains("'"+self.label+"'")]

            #if self.label == "Not relevant":
            #    self.dfs[key] = df[df["predicted_tag"].str.contains("'"+self.parent.label+"', 'Not relevant'")] # Make sure its the ones where this branch is not relevant
            #else:
            #    self.dfs[key] = df[df["predicted_tag"].str.contains("'"+self.label+"'")]
        

        for child in self.children:
            child.set_dfs(self.dfs)
    
    def print_info(self, fnc):
        try:
            print("\t"*self.depth, fnc(self))
        except StopIteration:
            return
        for child in self.children:
            child.print_info(fnc)


def populate(node, branch):
    if isinstance(branch, dict):
        for label, subtree in branch.items():
            child = Node(node, label)
            node.set_child(child)
            populate(child, subtree)
    elif isinstance(branch, list):
        for item in branch:
            if isinstance(item, dict):
                for label, subtree in item.items():
                    child = node.set_child(Node(node, label))
                    populate(child, subtree)
            else:
                node.set_child(Node(node, item))
    elif branch is None:
        return
    else:
        node.set_child(Node(node, branch))

