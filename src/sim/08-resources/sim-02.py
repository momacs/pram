'''
Resource transformation simulation modeling the following process: trees --> wood --> lumber
'''

import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from pram.entity import Group
from pram.rule   import Rule, TimeAlways
from pram.sim    import Simulation


# ----------------------------------------------------------------------------------------------------------------------
class TreeToLumberRule(Rule):
    '''
    Process modeled:  2 tree --> 1 wood --> 10 lumber
    '''

    STEP_1_TREE = 2.0
    STEP_1_WOOD = 1.0

    STEP_2_WOOD = 10.0
    STEP_2_LUMBER = 50.0

    def __init__(self):
        super().__init__('tree-to-lumber', TimeAlways())

    def apply(self, pop, group, iter, t):
        c = self.__class__              # shorthand
        q = group.get_attr('quantity')  # shorthand

        # Step 1: tree --> wood:
        if q['tree'] >= self.__class__.STEP_1_TREE:
            tree_pre   = float(q['tree'])
            tree_xform = c.STEP_1_TREE
            tree_post  = tree_pre - tree_xform

            wood_1_pre   = float(q['wood'])
            wood_1_xform = tree_xform / c.STEP_1_TREE * c.STEP_1_WOOD
            wood_1_post  = wood_1_pre + wood_1_xform

            # Step 2: wood --> lumber:
            if wood_1_post >= self.__class__.STEP_2_WOOD:
                wood_2_pre   = wood_1_post
                wood_2_xform = c.STEP_2_WOOD
                wood_2_post  = wood_2_pre - wood_2_xform

                lumber_pre   = float(q['lumber'])
                lumber_xform = wood_2_xform / c.STEP_2_WOOD * c.STEP_2_LUMBER
                lumber_post  = lumber_pre + lumber_xform

                q['tree']   = tree_post
                q['wood']   = wood_2_post
                q['lumber'] = lumber_post
                return None

            q['tree'] = tree_post
            q['wood'] = wood_1_post
            return None

        return None

    def is_applicable(self, group, iter, t):
        q = group.get_attr('quantity')  # shorthand

        return (
            super().is_applicable(group, iter, t) and
            group.has_attr({ 'type': 'resource' }) and
            q and (q.get('tree') is not None) and (q.get('wood') is not None) and (q.get('lumber') is not None)
        )


# ----------------------------------------------------------------------------------------------------------------------
(Simulation().
    set().
        pragma_autocompact(False).
        pragma_live_info(True).
        done().
    add().
        rule(TreeToLumberRule()).
        group(Group(n=1, attr={ 'type': 'resource', 'quantity': { 'tree': 100, 'wood': 0, 'lumber': 0 }})).
        done().
    # run(1).
    # run(9).
    # run(10).
    # run(20).
    # run(49).
    # run(50).
    # run(51).
    run(100).
    summary(False, 256,0,0,0, (1,0))
)
