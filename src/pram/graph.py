import graph_tool.all as gt
import math

from dotmap           import DotMap
from sortedcontainers import SortedSet


# ----------------------------------------------------------------------------------------------------------------------
class MassGraph(object):
    '''
    A graph of locus and flow of mass.

    This class holds the entire time-evolution of the group space and the associated mass flow.

    This class currently uses graph-tool library.  I've also considered using the Python-native NetworkX, but
    graph-tool is faster, more featurefull, and performance-transparent (i.e., all operations are annotated with the
    big O notation).  The only problem with graph-tool is that it is more difficult to install, which is going to be
    signifiant for users, especially the casual ones who just want to get feel for the PRAM package.  This points
    towards a conteinerized version of PRAM as possibly desirable.

    --------------------------------------------------------------------------------------------------------------------

    Good visualization options
        Sankey diagram
            https://www.data-to-viz.com/graph/sankey.html
            https://medium.com/@plotlygraphs/4-interactive-sankey-diagram-made-in-python-3057b9ee8616
            http://www.sankey-diagrams.com
        Chord diagram
            https://www.data-to-viz.com/graph/chord.html
        Edge bundling diagram
            https://www.data-to-viz.com/graph/edge_bundling.html
        Arc diagram
            https://www.data-to-viz.com/graph/arc.html
        Galleries
            https://observablehq.com/@vega

    --------------------------------------------------------------------------------------------------------------------

    TODO
        Performance
            https://stackoverflow.com/questions/36193773/graph-tool-surprisingly-slow-compared-to-networkx/36202660#36202660
        Other
            https://graph-tool.skewed.de/static/doc/draw.html
            https://graph-tool.skewed.de/static/doc/flow.html

    Ideas
        Determine groups involved in large mass dynamics and only show those
            Filter via range parameter (could be a statistic, e.g., IRQ)
        Automatically prune time ranges with little-to-no changes in mass dynamics
        Autocorrelation plot
        Calculate mass flow and summarize it statistically
        Use heatmap for group-group mass flow
            - A big heatmap with mass flow traveling in the direction of the diagonal is diagnostic of new groups
              being created and old ones not used and is a prime example of why autocompacting is a thing.
        Make the "vis" package to handle visualization
    '''

    def __init__(self):
        self.g = gt.Graph()

        # Vertex properties (graph-internal):
        self.vp = DotMap(
            iter = self.g.new_vp('int'),
            hash = self.g.new_vp('string'),
            mass = self.g.new_vp('double'),
            pos  = self.g.new_vp('vector<double>'),
            name = self.g.new_vp('string')
        )

        # Edge properties (graph-internal):
        self.ep = DotMap(
            iter   = self.g.new_ep('int'),
            mass   = self.g.new_ep('double'),
            name   = self.g.new_ep('string'),
            draw_w = self.g.new_ep('double')
        )

        self.group_hash_set = SortedSet()  # because groups can be removed via population compacting, we store all hashes (from the oldest to the newest)
        self.gg_flow = []                  # iteration-index group-group mass flow
        self.n_iter = -1

    def add_group(self, iter, hash, m):
        v = self.g.add_vertex()
        self.vp.iter[v] = iter
        self.vp.hash[v] = hash
        self.vp.mass[v] = m
        self.vp.name[v] = f'{m:.0f}'

        self.n_iter = max(self.n_iter, iter)

        return self

    def add_mass_flow(self, pop, iter, mass_flow_specs):
        self.gg_flow.append({})
        v_cnt = 0  # for calculating vertex y-coord

        for mtf in mass_flow_specs:
            self.gg_flow[iter][mtf.src.get_hash()] = {}

            # (1) Get the source group vertex:
            v_src = None
            v_src_lst = gt.find_vertex(self.g, self.vp.hash, mtf.src.get_hash())
            for v_src in v_src_lst:
                if self.vp.iter[v_src] == iter - 1:
                    break

            # (2) Add the destination group vertices:
            for g_dst in mtf.dst:
                v_dst = None

                # Check if the vertex has already been added for the current iteration:
                v_dst_lst = gt.find_vertex(self.g, self.vp.hash, g_dst.get_hash())
                if len(v_dst_lst) > 0:
                    v_dst = v_dst_lst[-1]
                    if self.vp.iter[v_dst] != iter:
                        v_dst = None
                else:
                    v_dst = None

                # Add the vertex:
                if v_dst is None:
                    m = pop.groups[g_dst.get_hash()].m  # get that from the population to get the appropriate mass for the current iteration

                    v_dst = self.g.add_vertex()
                    self.vp.iter[v_dst] = iter
                    self.vp.hash[v_dst] = g_dst.get_hash()
                    self.vp.mass[v_dst] = m
                    self.vp.pos [v_dst] = ((iter + 1) * 2, v_cnt * 2, 0)
                    self.vp.name[v_dst] = f'{m:.0f}'
                    v_cnt += 1

                # Add the edge:
                e = self.g.add_edge(v_src, v_dst)
                self.ep.iter  [e] = iter
                self.ep.mass  [e] = g_dst.m
                self.ep.name  [e] = f'{g_dst.m:.0f}'
                self.ep.draw_w[e] = math.log(g_dst.m, 2)

                # Remember the groups' hashes:
                self.group_hash_set.add(mtf.src.get_hash())
                self.group_hash_set.add(g_dst.get_hash())

                # Update the group-group flow structure:
                self.gg_flow[iter][mtf.src.get_hash()][g_dst.get_hash()] = g_dst.m

        return self

    def plot_states(self, size, filepath):
        # pos = gt.arf_layout(self.g, max_iter=10)
        self.set_pos()
        gt.graph_draw(
            self.g, pos=self.vp.pos,
            vertex_text=self.vp.name, vertex_size=100, vertex_pen_width=2, vertex_color=[0,0,0,1.0], vertex_fill_color=[209/255,242/255,255/255,1.0], vertex_text_color=[0,0,0,1.0], vertex_halo=False, vertex_font_size=20, vertex_font_family='sans-serif',
            # vertex_pie_fractions=[]
            edge_text=self.ep.name, edge_font_size=20, edge_pen_width=self.ep.draw_w, edge_font_family='sans-serif',
            output_size=size, output=filepath
        )
        return self

    def set_group_names(self, names):
        pass

    def set_pos(self):
        ''' Sets the position of every vertex.  That position is used for plotting the graph. '''

        v_cnt = 0
        for i in range(self.n_iter):
            for v in gt.find_vertex(self.g, self.vp.iter, i):
                self.vp.pos[v] = ((i + 1) * 2, v_cnt * 2, 0)
                v_cnt += 1

        return self
