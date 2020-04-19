'''
Static and dynamic rule analysis example.
'''

import pram.util as util

from pram.data   import GroupSizeProbe, ProbeMsgMode
from pram.entity import Group, GroupDBRelSpec, GroupQry, Site
from pram.rule   import GoToAndBackTimeAtRule, ResetSchoolDayRule, Rule, SEIRFluRule, TimeAlways, TimePoint
from pram.sim    import Simulation


# ----------------------------------------------------------------------------------------------------------------------
class AttrRule(Rule):
    def __init__(self, t=TimeAlways()):
        super().__init__('attr-rule', t)

    def apply(self, pop, group, iter, t):
        if iter > 10:  # dynamic rule analysis will capture the 'income' attribute below only after the 10th iteration
            if group.has_attr('income'):  # static rule analysis will always capture this
                pass


# ----------------------------------------------------------------------------------------------------------------------
rand_seed = 1928

(Simulation().
    set().
        rand_seed(rand_seed).
        pragma_autocompact(True).
        pragma_live_info(True).
        done().
    add().
        rule(AttrRule()).
        done().
    new_group(100).
        set_attr('income', 50).  # this will be found to be relevant by static analysis based on the AttrRule's body
        set_attr('age', 22).     # this will be found to be superfluous by dynamic analysis
        done().
    run(13).  # 10 or fewer for static analysis to work; more than 10 for dynamic analysis
    summary()
)


# ----------------------------------------------------------------------------------------------------------------------
# CREATE TABLE people2 AS SELECT p.*, h.hh_income AS income FROM people p INNER JOIN households h ON p.sp_hh_id = h.sp_id;
# ALTER TABLE people RENAME TO people2;
# ALTER TABLE people2 RENAME TO people;

# -- income in household
# SELECT p.school_id, COUNT(*) AS n, AVG(h.hh_income) AS m, MIN(h.hh_income) AS l, MAX(h.hh_income) AS u FROM households h
# INNER JOIN people p on p.sp_hh_id = h.sp_id
# WHERE p.school_id IS NOT NULL
# GROUP BY p.school_id
# --LIMIT 20
