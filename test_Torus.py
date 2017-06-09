import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import cm
from Torus import Torus


def draw_graph(t):
    """
    :type t: Torus
    """
    graph = t.graph
    positions = t.create_node_positions()
    edge_list = t.create_edge_list()
    intensities = t.edge_color_intensities()
    edge_labels = t.create_edge_labels()

    nx.draw_networkx_edge_labels(graph, positions, edge_labels, font_size=14)
    nx.draw_networkx(graph, positions, edgelist=edge_list, edge_color=intensities, edge_cmap=cm.Reds,
                     node_size=275, font_size=10, with_labels=False)
    plt.draw()


torus = Torus(5, 10, 12)
print "h_queues:\n%r" % torus.h_queues
print "v_queues:\n%r" % torus.v_queues
print "The total weight = %r" % torus.check_weights()
draw_graph(torus)
plt.show()
(phi, pi) = torus.minimizing()
print "Maximum: %r" % phi + " pi: %r" % str(pi)
print "h_queues:\n%r" % torus.h_queues
print "v_queues:\n%r" % torus.v_queues
print "The total weight = %r" % torus.check_weights()
draw_graph(torus)
plt.show()
