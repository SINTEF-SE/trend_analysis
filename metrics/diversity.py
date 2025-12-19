import math

def set_number_of_leaves(node):
    if len(node.children)==0:
        node.number_of_leaves = 1
    else:
        node.number_of_leaves = 0
        for child in node.children:
            set_number_of_leaves(child)
            node.number_of_leaves += child.number_of_leaves

def set_subtree_length(node):
    node.subtree_length = 1
    for child in node.children:
        set_subtree_length(child)
        node.subtree_length += child.subtree_length

def calc_absolute_diversity(node, threshold = None):
    diversity = {}    
    if threshold is None:
        threshold = {}
        for key in node.dfs:
            l = len(node.dfs[key])
            threshold[key] = l/node.number_of_leaves
        for key in threshold:
            diversity[key] = 0
    else: # only include its own diversity if not calculating rel div for this node
        for key in threshold:
            k = len(node.dfs[key])
            diversity[key] = min(threshold[key], len(node.dfs[key]))

    node.diversity_threshold = threshold


    for child in node.children:
        child_diversity = calc_absolute_diversity(child, threshold=threshold)
        for key in diversity:
            diversity[key] += child_diversity[key]
    #print("  "*node.depth, node.label, "\t", diversity)
    return diversity

def _calc_diversity_score(node):
    
    abs_diversity = calc_absolute_diversity(node)
    rel_diversity = {}

    max_diversity = {}
    for key in node.dfs:
        max_diversity[key] = ((node.subtree_length-1) * node.diversity_threshold[key]) # subtree_length includes current node - we only want to include its descendants

    for key in abs_diversity:
        if abs_diversity[key] == 0:
            rel_diversity[key] = 0
        else:
            rel_diversity[key] = abs_diversity[key]/max_diversity[key]
    node.diversity_score = rel_diversity
    #print("  "*node.depth, node.label, "\t\t", rel_diversity, abs_diversity, "\t---\t", [(key, len(node.dfs[key])) for key in node.dfs])

    for child in node.children:
        calc_diversity_score(child)

def calc_diversity_score(node):
    set_number_of_leaves(node)
    set_subtree_length(node)
    _calc_diversity_score(node)
