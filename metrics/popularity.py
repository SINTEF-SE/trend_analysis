
def _calc_popularity(node, full_dfs):
    node.popularity = {}
    for key in node.dfs:
        node.popularity[key] = len(node.dfs[key]) / len(full_dfs[key])
    for child in node.children:
        _calc_popularity(child, full_dfs)

def calc_popularity(node):
    _calc_popularity(node, node.dfs)

