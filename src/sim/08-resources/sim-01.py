'''
Resource transformation simulation modeling the following process: trees --> wood --> lumber
'''

import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from pram.data   import GroupSizeProbe, ProbeMsgMode, ProbePersistanceDB
from pram.entity import Group, GroupDBRelSpec, GroupQry, GroupSplitSpec, Site
from pram.rule   import Rule, TimeAlways
from pram.sim    import Simulation
from pram.util   import Time as TimeU


# ----------------------------------------------------------------------------------------------------------------------
class TreeToWoodRule(Rule):
    '''
    Process modeled:  2 tree --> 1 wood
    '''

    IN_TREE  = 2.0
    OUT_WOOD = 1.0

    def __init__(self):
        super().__init__('tree-to-wood', TimeAlways())

    def apply(self, pop, group, iter, t):
        c = self.__class__  # shorthand

        tree_pre   = float(group.get_attr('tree'))
        tree_xform = c.IN_TREE
        tree_post  = tree_pre - tree_xform
        # tree_post  = -tree_xform

        wood_pre   = float(group.get_attr('wood') or 0)
        wood_xform = tree_xform / c.IN_TREE * c.OUT_WOOD
        wood_post  = wood_pre + wood_xform

        return [GroupSplitSpec(p=1.00, attr_set={ 'tree': tree_post, 'wood': wood_post })]

    def is_applicable(self, group, iter, t):
        return (
            super().is_applicable(group, iter, t) and
            group.has_attr({ 'type': 'resource' }) and
            group.has_attr('tree') and group.get_attr('tree') >= self.__class__.IN_TREE
        )


# ----------------------------------------------------------------------------------------------------------------------
class WoodToLumberRule(Rule):
    '''
    Process modeled:  1 wood --> 10 lumber
    '''

    IN_WOOD    = 10.0
    OUT_LUMBER = 50.0

    def __init__(self):
        super().__init__('wood-to-lumber', TimeAlways())

    def apply(self, pop, group, iter, t):
        c = self.__class__  # shorthand

        wood_pre   = float(group.get_attr('wood'))
        wood_xform = c.IN_WOOD
        # wood_post  = wood_pre - wood_xform
        # wood_post  = -wood_xform
        wood_post  = -wood_xform if wood_pre > 0 else 0

        lumber_pre   = float(group.get_attr('lumber') or 0)
        lumber_xform = wood_xform / c.IN_WOOD * c.OUT_LUMBER
        lumber_post  = lumber_pre + lumber_xform

        return [GroupSplitSpec(p=1.00, attr_set={ 'wood': wood_post, 'lumber': lumber_post })]

    def is_applicable(self, group, iter, t):
        return (
            super().is_applicable(group, iter, t) and
            group.has_attr({ 'type': 'resource' }) and
            group.has_attr('wood') and group.get_attr('wood') >= self.__class__.IN_WOOD
        )


# ----------------------------------------------------------------------------------------------------------------------
(Simulation().
    set().
        pragma_autocompact(False).
        pragma_live_info(True).
        done().
    add().
        rule(TreeToWoodRule()).
        rule(WoodToLumberRule()).
        group(Group(n=1, attr={ 'type': 'resource', 'tree': 100 })).
        done().
    # run(1).
    # run(10).
    # run(11).
    # run(21).
    # run(50).
    # run(51).
    run(100).
    summary(False, 256,0,0,0, (1,0))
)
