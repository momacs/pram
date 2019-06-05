import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


from pram.data   import ProbeMsgMode, GroupSizeProbe
from pram.entity import AttrFluStage, Group, GroupQry, GroupSplitSpec
from pram.rule   import Rule, TimeAlways
from pram.sim    import Simulation


# ----------------------------------------------------------------------------------------------------------------------
class ProgressFluRule(Rule):
    def __init__(self, t=TimeAlways(), memo=None):
        super().__init__('progress-flu', t, memo)

    def apply(self, pop, group, iter, t):
        # A Dynamic Bayesian net with the initial state distribution:
        #
        #     flu: (1 0 0)
        #
        # and the transition model:
        #
        #           s     e     r
        #     s  0.95  0.00  0.10
        #     e  0.05  0.50  0
        #     r  0     0.50  0.90

        p = 0.05  # prob of infection

        if group.has_attr({ 'flu': 's' }):
            return [
                GroupSplitSpec(p=1 - p, attr_set={ 'flu': 's' }),
                GroupSplitSpec(p=p,     attr_set={ 'flu': 'e' }),
                GroupSplitSpec(p=0.00,  attr_set={ 'flu': 'r' })
            ]
        if group.has_attr({ 'flu': 'e' }):
            return [
                GroupSplitSpec(p=0.00, attr_set={ 'flu': 's' }),
                GroupSplitSpec(p=0.50, attr_set={ 'flu': 'e' }),
                GroupSplitSpec(p=0.50, attr_set={ 'flu': 'r' })
            ]
        if group.has_attr({ 'flu': 'r' }):
            return [
                GroupSplitSpec(p=0.10, attr_set={ 'flu': 's' }),
                GroupSplitSpec(p=0.00, attr_set={ 'flu': 'e' }),
                GroupSplitSpec(p=0.90, attr_set={ 'flu': 'r' })
            ]

    def is_applicable(self, group, iter, t):
        return super().is_applicable(group, iter, t) and group.has_attr([ 'flu' ])

    def setup(self, pop, group):
        return [
            GroupSplitSpec(p=1.0, attr_set={ 'flu': 's' })
        ]


# ----------------------------------------------------------------------------------------------------------------------
class ProgressFluIncomeRule(Rule):
    def __init__(self, t=TimeAlways(), memo=None):
        super().__init__('progress-flu-income', t, memo)

    def apply(self, pop, group, iter, t):
        # A Dynamic Bayesian net.
        #
        # Initial state distribution:
        #     flu: (1 0 0)
        #     income: (0.5 0.5)
        #
        # Transition model:
        #                 L                    M
        #           S     E     R        S     E     R
        #     S  0.90  0.00  0.20     0.95  0.00  0.10
        #     E  0.10  0.75  0        0.05  0.50  0
        #     R  0     0.25  0.80     0     0.50  0.90

        # Low income:
        if group.has_attr({ 'income': 'l' }):
            p = 0.10  # prob of infection

            # Susceptible:
            if group.has_attr({ 'flu': 's' }):
                return [
                    GroupSplitSpec(p=1 - p, attr_set={ 'flu': 's' }),
                    GroupSplitSpec(p=p,     attr_set={ 'flu': 'e' }),
                    GroupSplitSpec(p=0.00,  attr_set={ 'flu': 'r' })
                ]
            # Exposed:
            if group.has_attr({ 'flu': 'e' }):
                return [
                    GroupSplitSpec(p=0.00, attr_set={ 'flu': 's' }),
                    GroupSplitSpec(p=0.75, attr_set={ 'flu': 'e' }),
                    GroupSplitSpec(p=0.25, attr_set={ 'flu': 'r' })
                ]
            # Recovered:
            if group.has_attr({ 'flu': 'r' }):
                return [
                    GroupSplitSpec(p=0.20, attr_set={ 'flu': 's' }),
                    GroupSplitSpec(p=0.00, attr_set={ 'flu': 'e' }),
                    GroupSplitSpec(p=0.80, attr_set={ 'flu': 'r' })
                ]

        # Medium income:
        if group.has_attr({ 'income': 'm' }):
            p = 0.05  # prob of infection

            # Susceptible:
            if group.has_attr({ 'flu': 's' }):
                return [
                    GroupSplitSpec(p=1 - p, attr_set={ 'flu': 's' }),
                    GroupSplitSpec(p=p,     attr_set={ 'flu': 'e' }),
                    GroupSplitSpec(p=0.00,  attr_set={ 'flu': 'r' })
                ]
            # Exposed:
            if group.has_attr({ 'flu': 'e' }):
                return [
                    GroupSplitSpec(p=0.00, attr_set={ 'flu': 's' }),
                    GroupSplitSpec(p=0.50, attr_set={ 'flu': 'e' }),
                    GroupSplitSpec(p=0.50, attr_set={ 'flu': 'r' })
                ]
            # Recovered:
            if group.has_attr({ 'flu': 'r' }):
                return [
                    GroupSplitSpec(p=0.10, attr_set={ 'flu': 's' }),
                    GroupSplitSpec(p=0.00, attr_set={ 'flu': 'e' }),
                    GroupSplitSpec(p=0.90, attr_set={ 'flu': 'r' })
                ]

    def is_applicable(self, group, iter, t):
        return super().is_applicable(group, iter, t) and group.has_attr([ 'flu', 'income' ])

    def setup(self, pop, group):
        return [
            GroupSplitSpec(p=1.0, attr_set={ 'flu': 's' })
        ]


# ----------------------------------------------------------------------------------------------------------------------
def sim01(iter_cnt):
    sim = (
        Simulation().
        add([
            ProgressFluRule(),
            GroupSizeProbe.by_attr('flu', 'flu', ['s', 'e', 'r'], msg_mode=ProbeMsgMode.CUMUL),
            Group(n=1000)
        ]).
        run(iter_cnt)
    )
    print(sim.probes[0].get_msg())
    print()


# ----------------------------------------------------------------------------------------------------------------------
def sim02(iter_cnt):
    sim = (
        Simulation().
        add([
            ProgressFluIncomeRule(),
            GroupSizeProbe.by_attr('flu', 'flu', ['s', 'e', 'r'], msg_mode=ProbeMsgMode.CUMUL),
            Group(n=500, attr={ 'income': 'l' }),
            Group(n=500, attr={ 'income': 'm' })
        ]).
        run(iter_cnt)
    )
    print(sim.probes[0].get_msg())
    print()


# ----------------------------------------------------------------------------------------------------------------------
def sim03(iter_cnt):
    sim = (
        Simulation().
        add([
            ProgressFluIncomeRule(),
            GroupSizeProbe(
                'flu-income',
                [
                    GroupQry(attr={ 'income': 'l', 'flu': 's' }),
                    GroupQry(attr={ 'income': 'l', 'flu': 'e' }),
                    GroupQry(attr={ 'income': 'l', 'flu': 'r' }),
                    GroupQry(attr={ 'income': 'm', 'flu': 's' }),
                    GroupQry(attr={ 'income': 'm', 'flu': 'e' }),
                    GroupQry(attr={ 'income': 'm', 'flu': 'r' })
                ],
                msg_mode=ProbeMsgMode.CUMUL,
            ),
            Group(n=500, attr={ 'income': 'l' }),
            Group(n=500, attr={ 'income': 'm' })
        ]).
        run(iter_cnt)
    )
    print(sim.probes[0].get_msg())
    print()


# ----------------------------------------------------------------------------------------------------------------------
iter_cnt = 36

sim01(iter_cnt)
sim02(iter_cnt)
sim03(iter_cnt)
