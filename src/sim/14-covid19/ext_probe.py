from pram.data        import Probe, Var
from pram.entity      import GroupQry
from pram.util        import Size, Time

from ext_group_qry import gq_S, gq_E, gq_IA, gq_IS, gq_R


# ----------------------------------------------------------------------------------------------------------------------
class PopProbe(Probe):
    def __init__(self, persistence=None, do_print=True):
        self.consts = []
        self.vars = [
            Var('m',   'float'),
            Var('m_S', 'float'),
            Var('p_S', 'float'),
            Var('m_E', 'float'),
            Var('p_E', 'float'),
            Var('m_I', 'float'),
            Var('p_I', 'float'),
            Var('m_R', 'float'),
            Var('p_R', 'float'),
            Var('m_X', 'float'),
            Var('p_X', 'float'),
            Var('md_I', 'float')
        ]

        super().__init__('pop', persistence)

        self.do_print = do_print

    def get_pop_aggr(self):
        """ Calculate aggregates (currently, sums) of the four SEIR disease states (I = IA + IS), plus dead mass. """

        m_S, p_S = self.pop.get_groups_mass_and_prop(gq_S)
        m_E, p_E = self.pop.get_groups_mass_and_prop(gq_E)
        # m_I, p_I = self.pop.get_groups_mass_and_prop(GroupQry(cond=[lambda g: g.has_attr({ disease_name: 'IA' }) or g.has_attr({ disease_name: 'IS' })]))
        m_R, p_R = self.pop.get_groups_mass_and_prop(gq_R)

        m_IA, p_IA = self.pop.get_groups_mass_and_prop(gq_IA)
        m_IS, p_IS = self.pop.get_groups_mass_and_prop(gq_IS)
        m_I = m_IA + m_IS
        p_I = p_IA + p_IS

        m_X = self.pop.m_out
        p_X = self.pop.m_out / self.pop_m_init if (self.pop_m_init > 0) else 0

        # md_IA = self.pop.get_groups_mass(GroupQry(attr={ disease_name: 'IA' }), hist_delta=2)
        # md_IS = self.pop.get_groups_mass(GroupQry(attr={ disease_name: 'IS' }), hist_delta=2)
        # md_I  = md_IA + md_IS
        md_I = 0.0

        return {
            'm_S': m_S, 'p_S': p_S,
            'm_E': m_E, 'p_E': p_E,
            'm_I': m_I, 'p_I': p_I,
            'm_R': m_R, 'p_R': p_R,
            'm_X': m_X, 'p_X': p_X,
            'md_I': md_I
        }

    def get_loc_aggr(self):
        """ Calculate aggregates (currently, sums) of the locations students can be at. """

        m_home,   p_home   = self.pop.get_groups_mass_and_prop(GroupQry(cond=[lambda g: g.is_at_site_name('home'  )]))
        m_work,   p_work   = self.pop.get_groups_mass_and_prop(GroupQry(cond=[lambda g: g.is_at_site_name('work'  )]))
        m_school, p_school = self.pop.get_groups_mass_and_prop(GroupQry(cond=[lambda g: g.is_at_site_name('school')]))

        return {
            'm_home':   m_home,   'p_home':   p_home,
            'm_work':   m_work,   'p_work':   p_work,
            'm_school': m_school, 'p_school': p_school
        }

    def run(self, iter, t, traj_id):
        if iter is None:  # TODO: Create an easily understood method to check this.
            self.pop_m_init = self.pop.m
            time_of_day = '-'
            is_morning = False
        else:
            is_morning = not iter % 2
            time_of_day = 'm' if is_morning else 'e'

        pop = self.get_pop_aggr()
        loc = self.get_loc_aggr()

        if self.do_print:
            pop_col_cnt = 8  # number of columns to use for printing population sizes
            print(
                f'{iter if (iter is not None) else "-":>4} ' +
                f'{time_of_day} ' +
                f'pop: {self.pop.m:>{pop_col_cnt},.0f}   ' +
                f'x: {pop["m_X"]:>{pop_col_cnt},.0f}|{pop["p_X"] * 100:>3.0f}%   ' +
                f's: {pop["m_S"]:>{pop_col_cnt},.0f}|{pop["p_S"] * 100:>3.0f}%   ' +
                f'e: {pop["m_E"]:>{pop_col_cnt},.0f}|{pop["p_E"] * 100:>3.0f}%   ' +
                f'i: {pop["m_I"]:>{pop_col_cnt},.0f}|{pop["p_I"] * 100:>3.0f}%   ' +
                # f'id: {pop["md_I"]:>{pop_col_cnt},.0f}   ' +
                f'r: {pop["m_R"]:>{pop_col_cnt},.0f}|{pop["p_R"] * 100:>3.0f}%   ' +
                f'@home: {loc["p_home"] * 100:>3.0f}%   ' +
                f'@work: {loc["p_work"] * 100:>3.0f}%   ' +
                f'@school: {loc["p_school"] * 100:>3.0f}%   ' +
                f'groups: {len(self.pop.groups):>10,d}    ' +
                f'{Size.bytes2human(self.pop.sim.comp_hist.mem_iter[-1]):>7}  ' +
                f'{Time.tsdiff2human(self.pop.sim.comp_hist.t_iter[-1]):>12}'
            )

        if self.persistence and not is_morning:
            if iter is None:
                self.persistence.persist(self, [self.pop.m, pop['m_S'], pop['p_S'], pop['m_E'], pop['p_E'], pop['m_I'], pop['p_I'], pop['m_R'], pop['p_R'], pop['m_X'], pop['p_X'], pop['md_I']], -1, -1, traj_id)
            else:
                self.persistence.persist(self, [self.pop.m, pop['m_S'], pop['p_S'], pop['m_E'], pop['p_E'], pop['m_I'], pop['p_I'], pop['m_R'], pop['p_R'], pop['m_X'], pop['p_X'], pop['md_I']], iter, t, traj_id)
