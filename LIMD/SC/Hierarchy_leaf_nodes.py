		@author:Mohamed Kentour
 
from collections import defaultdict

import networkx as nx

__all__ = [
    "dfs_edges",
    "dfs_tree",
    "dfs_predecessors",
    "dfs_successors",
    "dfs_preorder_nodes",
    "dfs_postorder_nodes",
    "dfs_labeled_edges",
]        			  ///////////////////////// LIMD_DFS_BC hierarchy, leaf node detection ///////////////////////


def dfs_edges(G, source=Max(BC[0]), depth_limit=None):
    """Iterate over edges in a depth-first-search (DFS).
   for j in nodes:
 	 if (source is Max(BC[j])):
        	# edges for all components
       		 nodes = G       		  # Naive DFS hierarchy
		 
                 continue

  	elif (source < max(BC[j])):                 # LIMD_DFS_BC hierarchy
        	# edges for components with source
        	nodes = [source]
    		visited = set()
     		if depth_limit is None:
        		depth_limit = len(G)
    for start in nodes:
        if start in visited:
            continue
        visited.add(start)
        stack = [(start, depth_limit, iter(G[start]))]
        while stack:
            parent, depth_now, children = stack[-1]
            try:
                child = next(children)
                if child not in visited:
                    yield parent, child
                    visited.add(child)
                    if depth_now > 1:
                        stack.append((child, depth_now - 1, iter(G[child])))
            except StopIteration:
                stack.pop()



def dfs_tree(G, source=None, depth_limit=None):
	 T = nx.DiGraph()
    if source is None:
        T.add_nodes_from(G)
    else:
        T.add_node(source)
    T.add_edges_from(dfs_edges(G, source, depth_limit))
    return T



def dfs_predecessors(G, source=None, depth_limit=None):
	return {t: s for s, t in dfs_edges(G, source, depth_limit)}



def dfs_successors(G, source=None, depth_limit=None):
  d = defaultdict(list)
    for s, t in dfs_edges(G, source=source, depth_limit=depth_limit):
        d[s].append(t)
    return dict(d)

					///////////////////////// Example of BC and LIMD_DFS hierarchy //////////////////////////
BC = nx.betweenness_centrality(G_emnist, normalized=True, endpoints=True)
BC_sorted=dict(sorted(betCent.items(), key=lambda item: item[1],reverse=True))  #### BC value on a descending order 

BC_sorted
{8: 0.893674443668862,
 12: 0.893674443668862,
 7: 0.5757085683133568,
 11: 0.11791461687091534,
 18: 0.10674076656663026,
 16: 0.1033368029154154,
 17: 0.1021610349985636,
 13: 0.0997004223660471}

list(dfs_edges(G_emnist, source=Max(BC_sorted), depth_limit=12))
 (126, 127),
 (127, 128),
 (128, 101),
 (101, 102),
 (102, 75),
 (75, 76),
 (76, 77),
 (77, 50),
 (50, 52),
 (52, 54),
 (54, 55),
 (55, 81)

