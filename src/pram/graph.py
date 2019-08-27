import graph_tool.all as gt
import math

from dotmap           import DotMap
from sortedcontainers import SortedSet


# ----------------------------------------------------------------------------------------------------------------------
class MassGraph(object):
    '''
    A graph of locus and flow of mass.

    This class holds the entire time-evolution of the group space and the associated mass flow.

    This class uses graph-tool library.  I've also considered using the Python-native NetworkX, but graph-tool is
    faster, more featurefull, and performance-transparent (i.e., all operations are annotated with the big O notation).
    The only problem with graph-tool is that it is more difficult to install, which is going to be signifiant for
    users, especially the casual ones who just want to get feel for the software.  This makes a conteinerized version
    desirable.

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
        Incorporate autocorrelation
        Calculate mass flow and summarize it statistically
        Heatmap for group-group mass flow
            - A big heatmap with mass flow traveling in the direction of the diagonal is diagnostic of new groups
              being created and old ones not used and is a prime example of why autocompacting is a thing.
    '''

    def __init__(self):
        self.g = gt.Graph()

        # Vertex properties (graph-internal):
        self.vp = DotMap(
            iter = self.g.new_vp('int'),
            hash = self.g.new_vp('string'),
            m    = self.g.new_vp('double'),  # mass
            m_p  = self.g.new_vp('double'),  # ^ (proportion of)
            name = self.g.new_vp('string'),
            pos  = self.g.new_vp('vector<double>')
        )

        # Edge properties (graph-internal):
        self.ep = DotMap(
            iter = self.g.new_ep('int'),
            m    = self.g.new_ep('double'),  # mass being transfered
            m_p  = self.g.new_ep('double'),  # ^ (proportion of)
            name = self.g.new_ep('string'),
            w    = self.g.new_ep('double')  # width (for plotting)
        )

        self.group_hash_set = SortedSet()  # because groups can be removed via population compacting, we store all hashes (from the oldest to the newest)
        self.gg_flow = []                  # iteration-index group-group mass flow

        self.n_iter = -1

    def add_group(self, iter, hash, m, m_p):
        v = self.g.add_vertex()
        self.vp.iter[v] = iter
        self.vp.hash[v] = hash
        self.vp.m   [v] = m
        self.vp.m_p [v] = m_p

        self.n_iter = max(self.n_iter, iter)
        self.iter_v_max = 0  # max number of vertices within a single iteration (used for plotting)

        return self

    def add_mass_flow(self, iter, src_hash, dst_hash, m, m_p):
        # if len(self.gg_flow) < iter:  # this assumes continuous iteration (i.e., does not handle gaps in iteration numbers)
        #     self.gg_flow.append({ src_hash: {} })

        # (1) Get the source vertex:
        v_src = None
        v_src_lst = gt.find_vertex(self.g, self.vp.hash, src_hash)
        for v_src in v_src_lst:
            if self.vp.iter[v_src] == iter - 1:
                break

        # (2) Get the destination vertex:
        v_dst = None
        v_dst_lst = gt.find_vertex(self.g, self.vp.hash, dst_hash)
        for v_dst in v_dst_lst:
            if self.vp.iter[v_dst] == iter:
                break

        # (3) Add the edge:
        e = self.g.add_edge(v_src, v_dst)
        self.ep.iter[e] = iter
        self.ep.m   [e] = m
        self.ep.m_p [e] = m_p
        self.ep.w   [e] = math.log(m, 2)

        # (4) Remember the groups' hashes:
        # self.group_hash_set.add(src_hash)
        # self.group_hash_set.add(dst_hash)

        # (5) Update the group-group flow structure:
        # self.gg_flow[iter][mtf.src.get_hash()][g_dst.get_hash()] = g_dst.m

        return self

    def plot_mass_flow_time_series(self, scale=(1.00, 1.00), filepath=None, iter_range=(-1, -1), v_prop=False, e_prop=False):
        '''
        Because this method plots a time series which can be of arbitrary length, there is no figure size argument.
        Instead, scale factor for the width and height of the figure is expected and defaults to no scaling.  The
        method decided on the figure size automatically, based on the iteration range given and the maximum number of
        groups in any iteration from that range.
        '''

        # (1) Set iteration range:
        if iter_range[1] == -1:
            iter_range = (iter_range[0], self.n_iter)
        else:
            iter_range = (iter_range[0], min(iter_range[1], self.n_iter))

        # (2) Hide vertices based on iteration range (only if it's been restricted):
        if iter_range[0] > -1 or iter_range[1] < self.n_iter:
            vp_filter = self.g.new_vertex_property('bool');
            for v in self.g.vertices():
                vp_filter[v] = (iter_range[0] <= self.vp.iter[v] <= iter_range[1])
            g = gt.GraphView(self.g, vp_filter)
        else:
            g = self.g

        # (3) Enforce layout:
        # pos = gt.arf_layout(self.g, max_iter=10)
        self.set_pos(iter_range)

        # (4) Set vertex and edge names:
        for v in self.g.vertices():
            self.vp.name[v] = f'{self.vp.m_p[v]:.2f}' if v_prop else f'{self.vp.m[v]:.0f}'

        for e in self.g.edges():
            self.ep.name[e] = f'{self.ep.m_p[e]:.2f}' if e_prop else f'{self.ep.m[e]:.0f}'

        # (5) Set the most reasonable plot parameters:
        n_iter_plot = iter_range[1] - max(iter_range[0], 0) + 1 + (1 if iter_range[0] == -1 else 0)
        v_size = 100 # (iter_range[1] - max(iter_range[0], 0)) * 10
        v_pen_width = 2
        v_font_size = 20
        e_font_size = 20

        size = (round(v_size * (n_iter_plot * 2.0) * scale[0]), round(self.iter_v_max * 200 * scale[1]))

        # (6) Plot:
        gt.graph_draw(
            g, pos=self.vp.pos, output_size=size, output=filepath, fit_view=1.00, bg_color=[1,1,1,1],
            vertex_text=self.vp.name, vertex_size=v_size, vertex_pen_width=v_pen_width, vertex_color=[0,0,0,1], vertex_fill_color=[215/255, 255/255, 209/255, 1.0], vertex_text_color=[0,0,0,1.0], vertex_halo=False, vertex_font_size=v_font_size, vertex_font_family='sans-serif',
            edge_text=self.ep.name, edge_font_size=e_font_size, edge_pen_width=self.ep.w, edge_font_family='sans-serif'
            # vertex_pie_fractions=[]
        )

        return self

    def set_group_names(self, names):
        pass

    def set_pos(self, iter_range):
        ''' Sets the position of every vertex.  That position is used for plotting the graph. '''

        self.iter_v_max = 0
        for (i,iter) in enumerate(range(iter_range[0], iter_range[1] + 1)):
            v_cnt = 0
            for v in gt.find_vertex(self.g, self.vp.iter, iter):
                self.vp.pos[v] = (i * 1, v_cnt * 1, 0)
                v_cnt += 1
                self.iter_v_max = max(self.iter_v_max, v_cnt)

        return self
