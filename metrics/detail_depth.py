def calc_absolute_falloffs(self):
    """Absolute falloff is number of documents labeled with this node but not with any child node"""
    continuers = {}
    for key in self.dfs:
        continuers[key] = 0
        for child in self.children:
            continuers[key] += len(child.dfs[key])
    
    falloffs = {}
    for key in self.dfs:
        falloffs[key] = len(self.dfs[key]) - continuers[key]
    self.absolute_falloffs = falloffs
    


def calc_absolute_depth_score(self):
    """Absolute depth score is sum(depth_datio*absolute_falloff), summed over all descendants"""

    # Depth ratio is how deep the node is relative to max depth under this node
    depth_ratio = self.depth / self.max_depth

    # calc absolute depth score from falloffs of this node
    calc_absolute_falloffs(self)
    abs_depth_scores = {} 
    for key in self.dfs:
        abs_depth_scores[key] = self.absolute_falloffs[key] * depth_ratio
    
    # add abs_depth_score from children
    for child in self.children:
        calc_absolute_depth_score(child)
        if child.label == "Not relevant":
            continue
        for key in self.dfs:
            abs_depth_scores[key] += child.abs_depth_scores[key]
        
    self.abs_depth_scores = abs_depth_scores
    



def _calc_avg_depth(node):

    # avg depth
    lens = []
    node.avg_depth = {}
    for key in node.dfs:
        if len(node.dfs[key]) < 1:
            avg_depth = 0
            continue
        l = len(node.dfs[key])
        for child in node.children:
            if child.label == "Not relevant":
                #print(child.label)
                l -= len(child.dfs[key]) # this should not count
        node.avg_depth[key] = node.abs_depth_scores[key]/l

    for child in node.children:
        _calc_avg_depth(child)


def calc_avg_depth(node):
    calc_absolute_depth_score(node)
    _calc_avg_depth(node)
