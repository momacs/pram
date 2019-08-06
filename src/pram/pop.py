from attr   import attrs, attrib
from dotmap import DotMap

from .entity import Group, GroupQry, Resource, Site


# ----------------------------------------------------------------------------------------------------------------------
@attrs(slots=True)
class MassFlowSpec(object):
    '''
    A specification of mass flow from one source group to a (possibly single-element) list of destination groups.

    A list of objects of this class for one simulation iteration encodes the full picture of mass flow in th system.
    '''

    src: Group = attrib()
    dst: list  = attrib(factory=list)


# ----------------------------------------------------------------------------------------------------------------------
class Population(object):
    def __init__(self):
        self.agents = AgentPopulation()
        self.groups = GroupPopulation()


# ----------------------------------------------------------------------------------------------------------------------
class AgentPopulation(object):
    def gen_group_pop(self):
        '''
        Generates a population of groups based on the current agents population.

        This method provides a general interface between popular agent-based modeling packages (e.g., NetLogo) and
        PramPy.
        '''

        pass


# ----------------------------------------------------------------------------------------------------------------------
class GroupPopulation(object):
    def __init__(self, sim, do_keep_mass_flow_specs=False):
        self.sim = sim

        self.groups = {}
        self.sites = {}
        self.resources = {}

        self.mass = 0

        self.is_frozen = False  # the simulation freezes the population on first run

        self.last_iter = DotMap(    # the most recent iteration
            mass_flow_tot = 0,      # total mass transfered
            mass_flow_specs = None  # a list of MassFlowSpec objects (i.e., the full picture of mass flow)
        )

        self.do_keep_mass_flow_specs = do_keep_mass_flow_specs

    def __repr__(self):
        pass

    def __str__(self):
        pass

    def add_group(self, group):
        '''
        Add a group if it doesn't exist and update the size if it does. This method also adds all Site objects from
        the group's relations so there is no need for the user to do this manually.

        All groups added to the population become frozen to prevent the user from changing their attribute and relations
        directly; doing it via group splitting is the proper way.
        '''

        if not self.is_frozen:
            self.mass += group.m

        g = self.groups.get(group.get_hash())
        if g is not None:
            g.m += group.m
        else:
            self.groups[group.get_hash()] = group
            self.add_sites    ([v for (_,v) in group.get_rel().items() if isinstance(v, Site)])
            self.add_resources([v for (_,v) in group.get_rel().items() if isinstance(v, Resource)])
            group.freeze()

        return self

    def add_groups(self, groups):
        for g in groups:
            self.add_group(g)
        return self

    def add_resource(self, resource):
        h = resource.get_hash()
        if h in self.resources.keys():
            return

        self.resources[h] = resource
        # resource.set_pop(self)

        return self

    def add_resources(self, resources):
        for r in resources:
            self.add_resource(r)
        return self

    def add_site(self, site):
        h = site.get_hash()
        if h in self.sites.keys():
            return

        self.sites[h] = site
        site.set_pop(self)

        return self

    def add_sites(self, sites):
        for s in sites:
            self.add_site(s)
        return self

    def apply_rules(self, rules, iter, t, is_rule_setup=False, is_rule_cleanup=False, is_sim_setup=False):
        '''
        Iterates through groups and for each applies all rules (which is handled by the Group class).  The result of
        (possible) rules applications is a list of new groups the original group should be split into.  When all the
        groups have been processed in this way, and all resulting groups have been defined, those resulting groups are
        subsequently used for mass transfer (which updates existing groups creates new ones).  Note that "resulting" is
        different from "new" because a group might have been split into to resulting groups one or more already exists
        in the group population.  Therefore, not all resulting groups (local scope) need to be new (global scope).
        '''

        mass_flow_specs = []
        src_group_hashes = set()  # hashes of groups to be updated (a safeguard against resetting mass of unaffected groups)
        for g in self.groups.values():
            dst_groups_g = g.apply_rules(self, rules, iter, t, is_rule_setup, is_rule_cleanup, is_sim_setup)
            if dst_groups_g is not None:
                mass_flow_specs.append(MassFlowSpec(g, dst_groups_g))
                src_group_hashes.add(g.get_hash())

        if len(mass_flow_specs) == 0:  # no mass to transfer
            return self

        return self.transfer_mass(src_group_hashes, mass_flow_specs, iter, t)

    def compact(self):
        # Remove empty groups:
        self.groups = { k:v for k,v in self.groups.items() if v.m > 0 }

        return self

    def freeze(self):
        # [g.freeze() for g in self.groups.values()]
        # self.groups = { g.get_hash(): g for g in self.groups.values() }

        self.is_frozen = True
        if self.sim.traj is not None and not self.sim.timer.i > 0:  # we check timer not to save initial state of a simulation that's been run already
            self.sim.traj.save_state(None)

        return self

    def gen_agent_pop(self):
        ''' Generates a population of agents based on the current groups population. '''

        pass

    def get_group(self, qry=None):
        '''
        Returns the group with the all attributes and relations as specified; or None if such a group does not
        exist.

        qry: GroupQry
        '''

        qry = qry or GroupQry()
        return self.groups.get(Group.gen_hash(qry.attr, qry.rel))

    def get_group_cnt(self, only_non_empty=False):
        if only_non_empty:
            return len([g for g in self.groups.values() if g.m > 0])
        else:
            return len(self.groups)

    def get_groups(self, qry=None):
        '''
        Returns a list of groups that contain the attributes and relations specified in the query.  If the query is
        None, all groups are returned.

        qry: GroupQry
        '''

        # qry = qry or GroupQry()
        # attr_set = set(qry.attr.items())
        # rel_set  = set(qry.rel.items())
        #
        # ret = []
        # for g in self.groups.values():
        #     if (set(g.attr.items()) & attr_set == attr_set) and (set(g.rel.items()) & rel_set == rel_set):
        #         ret.append(g)
        #
        # return ret

        if qry is None:
            return self.groups.values()

        # return [g for g in self.groups.values() if (qry.attr.items() <= g.attr.items()) and (qry.rel.items() <= g.rel.items())]
        return [g for g in self.groups.values() if (qry.attr.items() <= g.attr.items()) and (qry.rel.items() <= g.rel.items()) and all([fn(g) for fn in qry.cond])]

    def get_next_group_name(self):
        return f'g.{len(self.groups)}'

    def get_site_cnt(self):
        return len(self.sites)

    def get_mass(self):
        # return sum([g.m for g in self.groups.values()])
        return self.mass

    def transfer_mass(self, src_group_hashes, mass_flow_specs, iter, t):
        '''
        Transfers the mass as described by the list of "destination" groups.  "Source" groups (i.e., those that
        participate in mass transfer) have their masses reset before the most-transfer mass is tallied up.
        '''

        m_flow_tot = 0  # total mass transfered

        # Reset the mass of the groups being updated:
        for h in src_group_hashes:
            self.groups[h].m = 0

        for mts in mass_flow_specs:
            for g01 in mts.dst:
                g02 = self.get_group(GroupQry(g01.attr, g01.rel))

                if g02 is not None:  # group already exists
                    g02.m += g01.m
                else:                # group not found
                    self.add_group(g01)

                m_flow_tot += g01.m

        # Notify sites of mass transfer:
        # for s in self.sites.values():
        #     s.invalidate_pop()  # TODO: Develop this further (AFAIR, unused ATM).

        # Save last iteration info:
        self.last_iter.m_flow_tot = m_flow_tot
        if self.do_keep_mass_flow_specs:
            self.last_iter.mass_flow_specs = mass_flow_specs

        # Save the trajectory state:
        if self.sim.traj is not None:
            self.sim.traj.save_state(mass_flow_specs)

        return self


# ----------------------------------------------------------------------------------------------------------------------
# class MassGraph(object):
#     '''
#     A graph of locus and flow of mass.
#
#     This class holds the entire time-evolution of the group space and the associated mass flow.
#
#     This class currently uses graph-tool library.  I've also considered using the Python-native NetworkX, but
#     graph-tool is faster, more featurefull, and performance-transparent (i.e., all operations are annotated with the
#     big O notation).  The only problem with graph-tool is that it is more difficult to install, which is going to be
#     signifiant for users, especially the casual ones who just want to get feel for the PRAM package.  This points
#     towards a conteinerized version of PRAM as possibly desirable.
#
#     --------------------------------------------------------------------------------------------------------------------
#
#     Good visualization options
#         Sankey diagram
#             https://www.data-to-viz.com/graph/sankey.html
#             https://medium.com/@plotlygraphs/4-interactive-sankey-diagram-made-in-python-3057b9ee8616
#             http://www.sankey-diagrams.com
#         Chord diagram
#             https://www.data-to-viz.com/graph/chord.html
#         Edge bundling diagram
#             https://www.data-to-viz.com/graph/edge_bundling.html
#         Arc diagram
#             https://www.data-to-viz.com/graph/arc.html
#         Galleries
#             https://observablehq.com/@vega
#
#     --------------------------------------------------------------------------------------------------------------------
#
#     TODO
#         Performance
#             https://stackoverflow.com/questions/36193773/graph-tool-surprisingly-slow-compared-to-networkx/36202660#36202660
#         Other
#             https://graph-tool.skewed.de/static/doc/draw.html
#             https://graph-tool.skewed.de/static/doc/flow.html
#
#     Ideas
#         Determine groups involved in large mass dynamics and only show those
#             Filter via range parameter (could be a statistic, e.g., IRQ)
#         Automatically prune time ranges with little-to-no changes in mass dynamics
#         Autocorrelation plot
#         Calculate mass flow and summarize it statistically
#         Use heatmap for group-group mass flow
#             - A big heatmap with mass flow traveling in the direction of the diagonal is diagnostic of new groups
#               being created and old ones not used and is a prime example of why autocompacting is a thing.
#         Make the "vis" package to handle visualization
#     '''
#
#     def __init__(self):
#         self.g = gt.Graph()
#
#         # Vertex properties (graph-internal):
#         self.vp = DotMap(
#             iter = self.g.new_vp('int'),
#             hash = self.g.new_vp('string'),
#             mass = self.g.new_vp('double'),
#             pos  = self.g.new_vp('vector<double>'),
#             name = self.g.new_vp('string')
#         )
#
#         # Edge properties (graph-internal):
#         self.ep = DotMap(
#             iter   = self.g.new_ep('int'),
#             mass   = self.g.new_ep('double'),
#             name   = self.g.new_ep('string'),
#             draw_w = self.g.new_ep('double')
#         )
#
#         self.group_hash_set = SortedSet()  # because groups can be removed via population compacting, we store all hashes (from the oldest to the newest)
#         self.gg_flow = []                  # iteration-index group-group mass flow
#
#         self.n_iter = -1
#         self.mass_max = 0
#
#     def capture_initial_state(self, pop):
#         for (i,g) in enumerate(pop.groups.values()):
#             v = self.g.add_vertex()
#             self.vp.iter[v] = -1
#             self.vp.hash[v] = g.get_hash()
#             self.vp.mass[v] = g.m
#             self.vp.pos [v] = (0, i * 2, 0)
#             self.vp.name[v] = f'{g.m:.0f}'
#
#             self.mass_max += g.m
#
#         return self
#
#     def capture_state(self, pop, mass_flow_specs, iter, t):
#         self.gg_flow.append({})
#         v_cnt = 0  # for vertex y-coord
#
#         for mtf in mass_flow_specs:
#             self.gg_flow[iter][mtf.src.get_hash()] = {}
#
#             # (1) Get the source group vertex:
#             v_src = None
#             v_src_lst = gt.find_vertex(self.g, self.vp.hash, mtf.src.get_hash())
#             for v_src in v_src_lst:
#                 if self.vp.iter[v_src] == iter - 1:
#                     break
#
#             # (2) Add the destination group vertices:
#             for g_dst in mtf.dst:
#                 v_dst = None
#
#                 # Check if the vertex has already been added for the current iteration:
#                 v_dst_lst = gt.find_vertex(self.g, self.vp.hash, g_dst.get_hash())
#                 if len(v_dst_lst) > 0:
#                     v_dst = v_dst_lst[-1]
#                     if self.vp.iter[v_dst] != iter:
#                         v_dst = None
#                 else:
#                     v_dst = None
#
#                 # Add the vertex:
#                 if v_dst is None:
#                     m = pop.groups[g_dst.get_hash()].m  # get that from the population to get the appropriate mass for the current iteration
#
#                     v_dst = self.g.add_vertex()
#                     self.vp.iter[v_dst] = iter
#                     self.vp.hash[v_dst] = g_dst.get_hash()
#                     self.vp.mass[v_dst] = m
#                     self.vp.pos [v_dst] = ((iter + 1) * 2, v_cnt * 2, 0)
#                     self.vp.name[v_dst] = f'{m:.0f}'
#                     v_cnt += 1
#
#                 # Add the edge:
#                 e = self.g.add_edge(v_src, v_dst)
#                 self.ep.iter  [e] = iter
#                 self.ep.mass  [e] = g_dst.m
#                 self.ep.name  [e] = f'{g_dst.m:.0f}'
#                 self.ep.draw_w[e] = math.log(g_dst.m, 2)
#
#                 # Remember the groups' hashes:
#                 self.group_hash_set.add(mtf.src.get_hash())
#                 self.group_hash_set.add(g_dst.get_hash())
#
#                 # Update the group-group flow structure:
#                 self.gg_flow[iter][mtf.src.get_hash()][g_dst.get_hash()] = g_dst.m
#
#         self.n_iter = iter
#
#         return self
#
#     def plot_heatmap(self, size, filepath):
#         # data = np.zeros((self.n_iter, self.n_group, self.n_group), dtype=float)
#
#         iter = 1
#         # data = np.array((len(self.group_hash_set), len(self.group_hash_set)))
#         # data = {}
#         data = []
#         for h_src in self.group_hash_set:
#             # data[h_src] = {}
#             for h_dst in self.group_hash_set:
#                 if self.gg_flow[iter] is not None and self.gg_flow[iter].get(h_src) is not None: # and self.gg_flow[iter].get(h_src).get(h_dst) is not None:
#                     # data[h_src][h_dst] = self.gg_flow[iter].get(h_src).get(h_dst)
#                     data.append({ 'x': h_src, 'y': h_dst, 'z': self.gg_flow[iter].get(h_src).get(h_dst) })
#
#         # print(data)
#         # return self
#
#         # c = alt.Chart(alt.Data(values=data)).mark_rect().encode(x='x:O', y='y:O', color='z:Q')
#         c = alt.Chart(alt.Data(values=data)).mark_rect().encode(x='x:O', y='y:O', color=alt.Color('z:Q', scale=alt.Scale(type='linear', range=['#bfd3e6', '#6e016b'])))
#         c.save(filepath, webdriver='firefox')
#
#     def plot_states(self, size, filepath):
#         # pos = gt.arf_layout(self.g, max_iter=10)
#         gt.graph_draw(
#             self.g, pos=self.vp.pos,
#             vertex_text=self.vp.name, vertex_size=100, vertex_pen_width=2, vertex_color=[0,0,0,1.0], vertex_fill_color=[209/255,242/255,255/255,1.0], vertex_text_color=[0,0,0,1.0], vertex_halo=False, vertex_font_size=20, vertex_font_family='sans-serif',
#             # vertex_pie_fractions=[]
#             edge_text=self.ep.name, edge_font_size=20, edge_pen_width=self.ep.draw_w, edge_font_family='sans-serif',
#             output_size=size, output=filepath
#         )
#
#         return self
#
#     def plot_streamgraph_group(self, size, filepath):
#         data = []
#
#         # for v in self.g.vertices():
#         #     data.append({ "group": self.vp.hash[v], "iter": i + 1, "mass": self.vp.mass[v] })
#
#         for i in range(-1, self.n_iter):
#             for v in gt.find_vertex(self.g, self.vp.iter, i):
#                 data.append({ "group": self.vp.hash[v], "iter": i + 1, "mass": self.vp.mass[v] })
#
#         c = alt.Chart(alt.Data(values=data), width=size[0], height=size[1]).mark_area().encode(
#             alt.X('iter:Q', axis=alt.Axis(domain=False, tickSize=0), scale=alt.Scale(domain=(0, self.n_iter))),
#             alt.Y('sum(mass):Q', stack='center', scale=alt.Scale(domain=(0, self.mass_max))),
#             alt.Color('group:N', scale=alt.Scale(scheme='category20b'))
#         )
#         c.configure_axis(grid=False)
#         c.configure_view(strokeWidth=0)
#         c.save(filepath, scale_factor=2.0, webdriver='firefox')
#
#     def set_group_names(self, names):
#         pass
