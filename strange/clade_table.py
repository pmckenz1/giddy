#!/usr/bin/env python


import numpy as np
import pandas as pd
import toytree
import toyplot



def get_subtree_table(tree, tree_table):
    """
    Table of support for subtrees in tree, meaning the topology under a 
    node in the test tree (tree in tree table) matches the subtree in 
    the input tree under that node idx.
    """
    trees = [toytree._rawtree(i) for i in tree_table.genealogy]
    subtree_table = pd.DataFrame({
        node.idx: np.zeros(tree_table.shape[0], dtype=int) 
        for node in tree.treenode.traverse("postorder") 
        if not node.is_leaf() and not node.is_root()
    })

    # get dictionary of nodes in full tree
    ndict = tree.get_node_dict(1, 1)

    # check subtrees from full tree against treelist
    for idx, tre in enumerate(trees):

        # check each clade in tree
        for nidx in subtree_table.columns:
            dist = (
                ndict[nidx].robinson_foulds(tre.treenode)[0]
            )
            mono = (
                tre.treenode.check_monophyly(ndict[nidx].get_leaf_names(), "name")[0]
            )
            if (dist == 0) and (mono is True):
                subtree_table.loc[idx, nidx] = 1
    return subtree_table


def get_dist_array(tree_table):
    """
    Get robinson foulds distances between all pairwise trees.
    """
    mtre = toytree.mtree(tree_table.genealogy)
    rarr = np.zeros((len(mtre.treelist), len(mtre.treelist)))
    for i in range(len(mtre.treelist)):
        for j in range(len(mtre.treelist)):
            rf = (
                mtre.treelist[i].treenode.robinson_foulds(
                    mtre.treelist[j].treenode)[0]
            )
            rarr[i, j] = rf
            rarr[j, i] = rf
    return rarr


def get_clades(tree):
    """
    Returns a dictionary with {idx: ['a', 'b', 'c'], ...} mapping node 
    idx labels to lists of tip names that are descendant. 

    If the input tree is unrooted...

    Used internally by _get_clade_table(). 
    """
    clades = {}
    for node in tree.treenode.traverse():
        clade = set(i.name for i in node.get_leaves())
        if len(clade) > 1 and len(clade) < tree.ntips:
            clades[node.idx] = clade
    return clades


def get_clade_table(tree, tree_table):
    """
    Returns a binary table showing which clade in each test tree (trees in 
    the tree table) is also found in the species tree (input tree).
    """
    # get clades in spp. tree
    clades = get_clades(tree)

    # make empty dataframe w/ column for each internal node
    clade_table = pd.DataFrame({
        node.idx: np.zeros(tree_table.shape[0], dtype=int) 
        for node in tree.treenode.traverse("postorder") 
        if not node.is_leaf() and not node.is_root()
    })

    # get clades for each tree block
    tarr = [
        get_clades(toytree._rawtree(tree_table.genealogy[idx]))
        for idx in tree_table.index
    ]

    # ask whether each clade in spp tree is in each test tree
    for node in tree.treenode.traverse():
        if not node.is_leaf() and not node.is_root():
            arr = np.array([clades[node.idx] in i.values() for i in tarr])
            clade_table[node.idx] = arr.astype(int)
    return clade_table


def map_node_colors(tree):
    ncolors = []
    for i in tree.get_node_values("idx"):
        if not i:
            ncolors.append("black")
        else:
            ncolors.append(next(toytree.icolors1))
    return ncolors



def tree_clades_slider_plot(tree, tree_table, clade_table):
    """
    Draw tree with corresponding clade table bars.
    """
    # setup canvas and two axes
    canvas = toyplot.Canvas(width=500, height=550)
    ax0 = canvas.cartesian(
        bounds=(75, 425, 50, 250)
    )
    ax1 = canvas.cartesian(
        bounds=(50, 450, 300, 500),
        xlabel="Genomic position (Mb)"
    )

    # get internal nodes
    ndict = tree.get_node_dict(1, 1)
    for node in sorted(ndict):
        ndict[node].color = next(toytree.icolors1)

    # add tree to the first axis
    tree.draw(
        axes=ax0,
        layout='down', 
        node_sizes=17,
        node_labels=tree.get_node_values("idx"),
        node_style={"stroke": "#262626", "fill-opacity": 0.9},
        node_colors=tree.get_node_values("color"),
        tip_labels=True,
    )

    # 
    base = 0

    # iterate over columns (internal sptree nodes)
    for col in sorted(clade_table.columns):
        ax1.rectangle(
            tree_table.start,
            tree_table.end,
            np.repeat(base, tree_table.shape[0]),
            clade_table.loc[:, col] + base,
            color=ndict[col].color,
            style={
                "stroke-width": 1,
                "stroke": ndict[col].color,
            },
            opacity=0.9,
        )

        # advance base y height
        base += 1

    # styling
    ax0.show = False
    ax1.y.ticks.locator = toyplot.locator.Explicit(
        locations=np.arange(0, tree.nnodes - tree.ntips - 1) + 0.5,
        labels=sorted(clade_table.columns),
    )
    ax1.x.ticks.locator = toyplot.locator.Explicit(
        locations=np.linspace(0, tree_table.end.max(), 10),
        labels=range(0, 10),
    )
    ax1.x.ticks.show = True
    ax1.y.ticks.labels.angle = -90
    return canvas, (ax0, ax1)




