from pram.entity import Group, GroupQry, GroupSplitSpec, Site
from pram.io.pop import PopulationLocation
from pram.rule   import Rule, SimRule

from ext_group_qry import gq_S, gq_E, gq_IA, gq_IS, gq_R
from ext_data      import disease_name


# ----------------------------------------------------------------------------------------------------------------------
class DailyBehaviorRule(Rule):
    """
    Model of daily human behavior.

    Args:
        default_dst_site (Site): Default destination site.
        p_home_IS (float): Probability of staying at home when in infected symptomatic (IS) state.

    UI:
        p_home_IS (type:float, mode:range, none:False, min:0.00, max:1.00, init:0.80)
    """

    def __init__(self, default_dst_site, p_home_IS):
        self.default_dst_site = default_dst_site
        self.p_home_IS = p_home_IS

        super().__init__(f'daily-behavior-{self.default_dst_site}', group_qry=GroupQry(cond=[lambda g: g.has_rel(self.default_dst_site)]))

    def apply(self, pop, group, iter, t):
        is_morning = not iter % 2  # TODO: Make this a simulation variable
        if is_morning:
            return self.apply_morning(pop, group, iter, t)
        else:
            return self.apply_evening(pop, group, iter, t)

    def apply_morning(self, pop, group, iter, t):
        if pop.sim.get_var('closedown'):
            return None

        if group.has_attr({ disease_name: 'IS' }):
            return [
                GroupSplitSpec(p=1.0 - self.p_home_IS, rel_set={ Site.AT: group.get_rel(self.default_dst_site) }),
                GroupSplitSpec(p=      self.p_home_IS)
            ]
        return [GroupSplitSpec(p=1.0, rel_set={ Site.AT: group.get_rel(self.default_dst_site) })]

    def apply_evening(self, pop, group, iter, t):
        return [GroupSplitSpec(p=1.0, rel_set={ Site.AT: group.get_rel('home') })]


# ----------------------------------------------------------------------------------------------------------------------
class DiseaseRule(Rule):
    def __init__(self, primary_E_site, r0, p_E_IA, p_IA_IS, p_IS_R, p_home_E, p_social_E, soc_dist_comp_young, soc_dist_comp_old, p_fat_by_age_group):
        if p_home_E + p_social_E > 1.0:
            raise ValueError('p_home_E + p_social_E cannot be greater than 1.')

        self.primary_E_site      = primary_E_site  # primary exposure site
        self.r0                  = r0
        self.p_E_IA              = p_E_IA
        self.p_IA_IS             = p_IA_IS
        self.p_IS_R              = p_IS_R
        self.p_home_E            = p_home_E
        self.p_social_E          = p_social_E
        self.soc_dist_comp_young = soc_dist_comp_young  # social distancing compliance (young people)
        self.soc_dist_comp_old   = soc_dist_comp_old    # social distancing compliance (old   people)
        self.p_fat_by_age_group  = p_fat_by_age_group

        super().__init__(f'disease-progress-{self.primary_E_site}', group_qry=GroupQry(cond=[lambda g: g.has_rel(self.primary_E_site)]))

    def apply(self, pop, group, iter, t):
        if group.has_attr({ disease_name: 'S' }):
            p_E = 0.0

            if group.is_at_site_name(self.primary_E_site):
                prop_IA = group.get_site_at().get_mass_prop(gq_IA)
                prop_IS = group.get_site_at().get_mass_prop(gq_IS)
                prop_I = prop_IA + prop_IS

                p_E = min(1.0, self.r0 * (prop_I))
            elif self.p_home_E > 0 or self.p_social_E > 0:
                prop_IA = group.get_site_at().get_mass_prop(gq_IA)
                prop_IS = group.get_site_at().get_mass_prop(gq_IS)
                prop_I = prop_IA + prop_IS

                if group.get_attr('age_group') == '0-50':
                    soc_dist_comp = self.soc_dist_comp_young
                else:
                    soc_dist_comp = self.soc_dist_comp_old

                p_E = min(1.0, self.r0 * (prop_I))
                p_E = p_E * self.p_home_E + p_E * self.p_social_E * (1.0 - soc_dist_comp)

            if p_E == 0:  # nothing to split (TODO: Can this be worked into PyPRAM?)
                return None
            else:
                return [
                    GroupSplitSpec(p=1 - p_E, attr_set={ disease_name: 'S' }),
                    GroupSplitSpec(p=    p_E, attr_set={ disease_name: 'E' })
                ]

        if group.has_attr({ disease_name: 'E' }):
            return [
                GroupSplitSpec(p=1 - self.p_E_IA, attr_set={ disease_name: 'E'  }),
                GroupSplitSpec(p=    self.p_E_IA, attr_set={ disease_name: 'IA' }),
            ]

        if group.has_attr({ disease_name: 'IA' }):
            return [
                GroupSplitSpec(p=1 - self.p_IA_IS, attr_set={ disease_name: 'IA' }),
                GroupSplitSpec(p=    self.p_IA_IS, attr_set={ disease_name: 'IS' })
            ]

        if group.has_attr({ disease_name: 'IS' }):
            p_fat = self.p_fat_by_age_group.get(group.get_attr('age_group'), 0.0)
            if p_fat is None:
                raise ValueError(f'Unexpected age group: {age_group}')

            return [
                GroupSplitSpec(p=                  p_fat, attr_set=Group.VOID),  # TODO: Make this a special class
                GroupSplitSpec(p=1 - self.p_IS_R,         attr_set={ disease_name: 'IS' }),
                GroupSplitSpec(p=    self.p_IS_R - p_fat, attr_set={ disease_name: 'R'  })
            ]


# ----------------------------------------------------------------------------------------------------------------------
class DiseaseRule2(Rule):
    def __init__(self, primary_E_site, sei2r_params, p_home_E, p_social_E, soc_dist_comp_young, soc_dist_comp_old, p_fat_by_age_group):
        if p_home_E + p_social_E > 1.0:
            raise ValueError('p_home_E + p_social_E cannot be greater than 1.')

        self.primary_E_site      = primary_E_site  # primary exposure site
        self.sei2r_params        = sei2r_params
        self.p_home_E            = p_home_E
        self.p_social_E          = p_social_E
        self.soc_dist_comp_young = soc_dist_comp_young  # social distancing compliance (young people)
        self.soc_dist_comp_old   = soc_dist_comp_old    # social distancing compliance (old   people)
        self.p_fat_by_age_group  = p_fat_by_age_group

        super().__init__(f'disease-progress-{self.primary_E_site}', group_qry=GroupQry(cond=[lambda g: g.has_rel(self.primary_E_site)]))

    def apply(self, pop, group, iter, t):
        if group.has_attr({ disease_name: 'S' }):
            p_E = 0.0

            if group.is_at_site_name(self.primary_E_site):
                prop_IA = group.get_site_at().get_mass_prop(gq_IA)
                prop_IS = group.get_site_at().get_mass_prop(gq_IS)
                prop_I = prop_IA + prop_IS

                p_E = min(1.0, self.sei2r_params.r0 * (prop_I))
            elif self.p_home_E > 0 or self.p_social_E > 0:
                prop_IA = group.get_site_at().get_mass_prop(gq_IA)
                prop_IS = group.get_site_at().get_mass_prop(gq_IS)
                prop_I = prop_IA + prop_IS

                if group.get_attr('age_group') == '0-50':
                    soc_dist_comp = self.soc_dist_comp_young
                else:
                    soc_dist_comp = self.soc_dist_comp_old

                p_E = min(1.0, self.sei2r_params.r0 * (prop_I))
                p_E = p_E * self.p_home_E + p_E * self.p_social_E * (1.0 - soc_dist_comp)

            if p_E == 0:  # nothing to split (TODO: Can this be worked into PyPRAM?)
                return None
            else:
                return [
                    GroupSplitSpec(p=1 - p_E, attr_set={ disease_name: 'S' }),
                    GroupSplitSpec(p=    p_E, attr_set={ disease_name: 'E' })
                ]

        if group.has_attr({ disease_name: 'E' }):
            return [
                GroupSplitSpec(p=1 - self.sei2r_params.kappa_1, attr_set={ disease_name: 'E'  }),
                GroupSplitSpec(p=    self.sei2r_params.kappa_1, attr_set={ disease_name: 'IA' }),
            ]

        if group.has_attr({ disease_name: 'IA' }):
            return [
                GroupSplitSpec(p=1 - self.sei2r_params.kappa_2, attr_set={ disease_name: 'IA' }),
                GroupSplitSpec(p=    self.sei2r_params.kappa_2, attr_set={ disease_name: 'IS' })
            ]

        if group.has_attr({ disease_name: 'IS' }):
            p_fat = self.p_fat_by_age_group.get(group.get_attr('age_group'), 0.0)
            if p_fat is None:
                raise ValueError(f'Unexpected age group: {age_group}')

            return [
                GroupSplitSpec(p=                              p_fat, attr_set=Group.VOID),  # TODO: Make this a special class
                GroupSplitSpec(p=1 - self.sei2r_params.gamma,         attr_set={ disease_name: 'IS' }),
                GroupSplitSpec(p=    self.sei2r_params.gamma - p_fat, attr_set={ disease_name: 'R'  })
            ]


# ----------------------------------------------------------------------------------------------------------------------
class DiseaseRuleMobility(Rule):
    def __init__(self, primary_E_site, r0, p_E_IA, p_IA_IS, p_IS_R, p_home_E, soc_dist_comp_young, soc_dist_comp_old, p_fat_by_age_group, pop_mobility=None, p_social_E_max=1.0):
        self.primary_E_site      = primary_E_site       # primary exposure site
        self.r0                  = r0
        self.p_E_IA              = p_E_IA
        self.p_IA_IS             = p_IA_IS
        self.p_IS_R              = p_IS_R
        self.p_home_E            = p_home_E
        self.soc_dist_comp_young = soc_dist_comp_young  # social distancing compliance (young people)
        self.soc_dist_comp_old   = soc_dist_comp_old    # social distancing compliance (old   people)
        self.p_social_E_max      = p_social_E_max
        self.p_fat_by_age_group  = p_fat_by_age_group

        if isinstance(pop_mobility, str):  # pop_mobility is assumed to contain database file path
            self.pop_mobility = PopulationLocation(pop_mobility)
            self.pop_mobility.set_mobility_first_day_of_year(61)
            self.pop_mobility.set_contacts_first_day_of_year(61)
        else:
            self.pop_mobility = pop_mobility

        super().__init__(f'disease-progress-mobility-{self.primary_E_site}', group_qry=GroupQry(cond=[lambda g: g.has_rel(self.primary_E_site)]))

    def apply(self, pop, group, iter, t):
        if group.has_attr({ disease_name: 'S' }):
            p_E = 0.0

            if group.is_at_site_name(self.primary_E_site):
                prop_IA = group.get_site_at().get_mass_prop(gq_IA)
                prop_IS = group.get_site_at().get_mass_prop(gq_IS)
                prop_I = prop_IA + prop_IS

                p_E = min(1.0, self.r0 * (prop_I))
            else:
                if isinstance(self.pop_mobility, PopulationLocation):
                    day = iter // 2 + 1  # ordinal number of day from the start of simulation (two iterations per day)
                    p_social_E = min(self.pop_mobility.get_contacts_by_day_of_year(group.get_rel('home').name, 2020, day, True) or 0 / 100, self.p_social_E_max)
                elif isinstance(self.pop_mobility, float):
                    p_social_E = min(self.pop_mobility, self.p_social_E_max)

                prop_IA = group.get_site_at().get_mass_prop(gq_IA)
                prop_IS = group.get_site_at().get_mass_prop(gq_IS)
                prop_I = prop_IA + prop_IS

                if group.get_attr('age_group') == '0-50':
                    soc_dist_comp = self.soc_dist_comp_young
                else:
                    soc_dist_comp = self.soc_dist_comp_old

                p_E = min(1.0, self.r0 * (prop_I))
                p_E = p_E * self.p_home_E + p_E * p_social_E * (1.0 - soc_dist_comp)

            if p_E == 0:  # nothing to split (TODO: Can this be worked into PyPRAM?)
                return None
            else:
                return [
                    GroupSplitSpec(p=1 - p_E, attr_set={ disease_name: 'S' }),
                    GroupSplitSpec(p=    p_E, attr_set={ disease_name: 'E' })
                ]

        if group.has_attr({ disease_name: 'E' }):
            return [
                GroupSplitSpec(p=1 - self.p_E_IA, attr_set={ disease_name: 'E'  }),
                GroupSplitSpec(p=    self.p_E_IA, attr_set={ disease_name: 'IA' }),
            ]

        if group.has_attr({ disease_name: 'IA' }):
            return [
                GroupSplitSpec(p=1 - self.p_IA_IS, attr_set={ disease_name: 'IA' }),
                GroupSplitSpec(p=    self.p_IA_IS, attr_set={ disease_name: 'IS' })
            ]

        if group.has_attr({ disease_name: 'IS' }):
            p_fat = self.p_fat_by_age_group.get(group.get_attr('age_group'), 0.0)
            if p_fat is None:
                raise ValueError(f'Unexpected age group: {age_group}')

            return [
                GroupSplitSpec(p=                  p_fat, attr_set=Group.VOID),  # TODO: Make this a special class
                GroupSplitSpec(p=1 - self.p_IS_R,         attr_set={ disease_name: 'IS' }),
                GroupSplitSpec(p=    self.p_IS_R - p_fat, attr_set={ disease_name: 'R'  })
            ]


# ----------------------------------------------------------------------------------------------------------------------
class DiseaseRuleMobility2(Rule):
    def __init__(self, primary_E_site, sei2r_params, p_home_E, soc_dist_comp_young, soc_dist_comp_old, p_fat_by_age_group, pop_mobility=None, p_social_E_max=1.0):
        self.primary_E_site      = primary_E_site       # primary exposure site
        self.sei2r_params        = sei2r_params
        self.p_home_E            = p_home_E
        self.soc_dist_comp_young = soc_dist_comp_young  # social distancing compliance (young people)
        self.soc_dist_comp_old   = soc_dist_comp_old    # social distancing compliance (old   people)
        self.p_social_E_max      = p_social_E_max
        self.p_fat_by_age_group  = p_fat_by_age_group

        if isinstance(pop_mobility, str):  # pop_mobility is assumed to contain database file path
            self.pop_mobility = PopulationLocation(pop_mobility)
            self.pop_mobility.set_mobility_first_day_of_year(61)
            self.pop_mobility.set_contacts_first_day_of_year(61)
        else:
            self.pop_mobility = pop_mobility

        super().__init__(f'disease-progress-mobility-{self.primary_E_site}', group_qry=GroupQry(cond=[lambda g: g.has_rel(self.primary_E_site)]))

    def apply(self, pop, group, iter, t):
        if group.has_attr({ disease_name: 'S' }):
            p_E = 0.0

            if group.is_at_site_name(self.primary_E_site):
                prop_IA = group.get_site_at().get_mass_prop(gq_IA)
                prop_IS = group.get_site_at().get_mass_prop(gq_IS)
                prop_I = prop_IA + prop_IS

                p_E = min(1.0, self.sei2r_params.r0 * (prop_I))
            else:
                if isinstance(self.pop_mobility, PopulationLocation):
                    day = iter // 2 + 1  # ordinal number of day from the start of simulation (two iterations per day)
                    p_social_E = min(self.pop_mobility.get_contacts_by_day_of_year(group.get_rel('home').name, 2020, day, True) or 0 / 100, self.p_social_E_max)
                elif isinstance(self.pop_mobility, float):
                    p_social_E = min(self.pop_mobility, self.p_social_E_max)

                prop_IA = group.get_site_at().get_mass_prop(gq_IA)
                prop_IS = group.get_site_at().get_mass_prop(gq_IS)
                prop_I = prop_IA + prop_IS

                if group.get_attr('age_group') == '0-50':
                    soc_dist_comp = self.soc_dist_comp_young
                else:
                    soc_dist_comp = self.soc_dist_comp_old

                p_E = min(1.0, self.sei2r_params.r0 * (prop_I))
                p_E = p_E * self.p_home_E + p_E * p_social_E * (1.0 - soc_dist_comp)

            if p_E == 0:  # nothing to split (TODO: Can this be worked into PyPRAM?)
                return None
            else:
                return [
                    GroupSplitSpec(p=1 - p_E, attr_set={ disease_name: 'S' }),
                    GroupSplitSpec(p=    p_E, attr_set={ disease_name: 'E' })
                ]

        if group.has_attr({ disease_name: 'E' }):
            return [
                GroupSplitSpec(p=1 - self.sei2r_params.kappa_1, attr_set={ disease_name: 'E'  }),
                GroupSplitSpec(p=    self.sei2r_params.kappa_1, attr_set={ disease_name: 'IA' }),
            ]

        if group.has_attr({ disease_name: 'IA' }):
            return [
                GroupSplitSpec(p=1 - self.sei2r_params.kappa_2, attr_set={ disease_name: 'IA' }),
                GroupSplitSpec(p=    self.sei2r_params.kappa_2, attr_set={ disease_name: 'IS' })
            ]

        if group.has_attr({ disease_name: 'IS' }):
            p_fat = self.p_fat_by_age_group.get(group.get_attr('age_group'), 0.0)
            if p_fat is None:
                raise ValueError(f'Unexpected age group: {age_group}')

            return [
                GroupSplitSpec(p=                              p_fat, attr_set=Group.VOID),  # TODO: Make this a special class
                GroupSplitSpec(p=1 - self.sei2r_params.gamma,         attr_set={ disease_name: 'IS' }),
                GroupSplitSpec(p=    self.sei2r_params.gamma - p_fat, attr_set={ disease_name: 'R'  })
            ]


# ----------------------------------------------------------------------------------------------------------------------
class ClosedownIntervention(SimRule):
    def __init__(self, prop_pop_IS_threshold):
        super().__init__('closedown-intervention')

        self.vars = { 'closedown': False }
        self.prop_pop_IS_threshold = prop_pop_IS_threshold

    def apply(self, sim, iter, t):
        if not sim.get_var('closedown'):
            prop_inf = sim.pop.get_groups_mass_prop(gq_IS)
            if prop_inf >= self.prop_pop_IS_threshold:
                sim.set_var('closedown', True)


# ----------------------------------------------------------------------------------------------------------------------
class ReopenIntervention(SimRule):
    def __init__(self, prop_pop_IS_threshold):
        super().__init__('reopen-intervention')

        self.vars = { 'closedown': False }
        self.prop_pop_IS_threshold = prop_pop_IS_threshold

    def apply(self, sim, iter, t):
        if sim.get_var('closedown'):
            prop_inf = sim.pop.get_groups_mass_prop(gq_IS)
            if prop_inf <= self.prop_pop_IS_threshold:
                sim.set_var('closedown', False)
