import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from pram.data   import GroupSizeProbe, ProbeMsgMode
from pram.entity import Group, GroupQry, Site
from pram.rule   import SimpleFluLocationRule, DiscreteVarMarkovChain
from pram.sim    import Simulation


# ----------------------------------------------------------------------------------------------------------------------
def get_probe_flu_at(school, name=None):
    return GroupSizeProbe(
        name=name or school.name,
        queries=[
            GroupQry(attr={ 'flu': 's' }, rel={ 'school': school }),
            GroupQry(attr={ 'flu': 'i' }, rel={ 'school': school }),
            GroupQry(attr={ 'flu': 'r' }, rel={ 'school': school })
        ],
        qry_tot=GroupQry(rel={ 'school': school }),
        msg_mode=ProbeMsgMode.DISP
    )


# ----------------------------------------------------------------------------------------------------------------------
def get_flu_s_sv(pop, group, iter, t):
    '''
    Returns the stochastic vector for the 's' state of the 'flu' attribute.  That vector becomes one of the rows of the
    transition matrix of the underlying discrete-time non-homogenous Markov chain.
    '''

    at  = group.get_rel(Site.AT)
    n   = at.get_pop_size()                               # total    population at the group's current location
    n_i = at.get_pop_size(GroupQry(attr={ 'flu': 'i' }))  # infected population at the group's current location

    p_inf = float(n_i) / float(n)  # changes every iteration (i.e., the source of the simulation dynamics)

    return [1 - p_inf, p_inf, 0.00]


# ----------------------------------------------------------------------------------------------------------------------
home     = Site('home')
school_l = Site('school-l')
school_m = Site('school-m')


(Simulation().
    set().
        pragma_analyze(False).
        pragma_autocompact(True).
        # pragma_live_info(True).
        done().
    add([
        SimpleFluLocationRule(),
        DiscreteVarMarkovChain('flu', { 's': get_flu_s_sv, 'i': [0.00, 0.80, 0.20], 'r': [0.10, 0.00, 0.90] }),
        get_probe_flu_at(school_l),
        Group('g1', 450, attr={ 'flu': 's', 'sex': 'f', 'income': 'm', 'pregnant': 'no', 'mood': 'happy'   }, rel={ Site.AT: school_m, 'school': school_m, 'home': home}),
        Group('g2',  50, attr={ 'flu': 'i', 'sex': 'f', 'income': 'm', 'pregnant': 'no', 'mood': 'annoyed' }, rel={ Site.AT: school_m, 'school': school_m, 'home': home}),
        Group('g3', 450, attr={ 'flu': 's', 'sex': 'm', 'income': 'm', 'pregnant': 'no', 'mood': 'happy'   }, rel={ Site.AT: school_m, 'school': school_m, 'home': home}),
        Group('g4',  50, attr={ 'flu': 'i', 'sex': 'm', 'income': 'm', 'pregnant': 'no', 'mood': 'annoyed' }, rel={ Site.AT: school_m, 'school': school_m, 'home': home}),
        Group('g5', 450, attr={ 'flu': 's', 'sex': 'f', 'income': 'l', 'pregnant': 'no', 'mood': 'happy'   }, rel={ Site.AT: school_l, 'school': school_l, 'home': home}),
        Group('g6',  50, attr={ 'flu': 'i', 'sex': 'f', 'income': 'l', 'pregnant': 'no', 'mood': 'annoyed' }, rel={ Site.AT: school_l, 'school': school_l, 'home': home}),
        Group('g7', 450, attr={ 'flu': 's', 'sex': 'm', 'income': 'l', 'pregnant': 'no', 'mood': 'happy'   }, rel={ Site.AT: school_l, 'school': school_l, 'home': home}),
        Group('g8',  50, attr={ 'flu': 'i', 'sex': 'm', 'income': 'l', 'pregnant': 'no', 'mood': 'annoyed' }, rel={ Site.AT: school_l, 'school': school_l, 'home': home})
    ]).
    run(5)
)
