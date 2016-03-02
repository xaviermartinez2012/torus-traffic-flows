import math
import random
# NetworkX python package is used to create the digraph
import networkx as nx


# Torus.py ################################################
# Designed by Xavier Martinez #############################
# for the HSI-STEM Winter Research Experience 2016 ########
# #########################################################


class Torus:
    def __init__(self, n, k, psi):
        assert (n > 1), "n=%r is too small a value\n" % n
        assert (k > 0), "k=%r is too small a value\n" % k
        assert type(n) is int, "n=%r is not an int value" % n
        assert type(k) is int, "k=%r is not an int value" % k
        assert type(psi) is int, "psi=%r is not an int value" % psi
        assert ((psi % 2) == 0), "psi=%r is not even" % psi

        random.seed(8)

        self.n = n
        self.size = n * n
        self.k = k
        self.psi = psi
        self.number_of_agents = self.k * self.n

        self.h_queues = {}
        self.v_queues = {}
        self.h_g_times = {}
        self.v_g_times = {}
        self.shifts = []

        self.graph = nx.DiGraph()
        self.looping_edges = []
        self.graph.add_nodes_from(range(0, self.size), h_queue=0, v_queue=0, h_g_time=0, v_g_time=0)
        self.create_graph()
        self.initialize()

    def initialize(self):
        for x in xrange(0, self.size):
            self.shifts.append(0)
        self.assign_queues()
        self.apply_green_time()

    def reset_shifts(self):
        for x in xrange(0, self.size):
            self.shifts[x] = 0

    def assign_queues(self):
        h_queues = []
        for x in xrange(0, self.n):
            h_queues.append(self.agents())
        row_counter = 0
        counter = self.n - 1

        for x in xrange(0, self.size):
            if x != counter:
                self.graph.node[x]['h_queue'] = h_queues[self.row_number(x)][row_counter]
            else:
                self.graph.node[x]['h_queue'] = h_queues[self.row_number(x)][row_counter]
                counter += self.n
                row_counter += 1
        del h_queues

        v_queues = []
        for x in xrange(0, self.n):
            v_queues.append(self.agents())
        column_counter = 0
        counter = self.n - 1

        for x in xrange(0, self.size):
            if x != counter:
                self.graph.node[x]['v_queue'] = v_queues[self.column_number(x)][column_counter]
                column_counter += 1
            else:
                self.graph.node[x]['v_queue'] = v_queues[self.column_number(x)][column_counter]
                counter += self.n
                column_counter = 0
        del v_queues

        self.h_queues = nx.get_node_attributes(self.graph, "h_queue")
        self.v_queues = nx.get_node_attributes(self.graph, "v_queue")

    def agents(self):
        agent_list = []
        for x in xrange(0, self.n):
            agent_list.append(self.k)
        agent_list = self.permutation(agent_list)
        return agent_list

    def permutation(self, k_list):
        for i in xrange(0, self.n * self.n * self.n):
            a = random.randint(0, self.n - 1)
            b = random.randint(0, self.n - 1)
            # Make sure that k_list[b] is 0 or greater
            if k_list[b] >= 1:
                k_list[b] -= 1
                k_list[a] += 1
        return k_list

    def apply_green_time(self):
        for x in xrange(0, self.size):
            self.graph.node[x]['h_g_time'] = int(self.shifts[x] + (self.psi / 2))
            self.graph.node[x]['v_g_time'] = int((self.psi / 2) - self.shifts[x])
        self.h_g_times = nx.get_node_attributes(self.graph, "h_g_time")
        self.v_g_times = nx.get_node_attributes(self.graph, "v_g_time")

    def create_graph(self):
        # The counter variable is used to determine when to change the direction of the edge
        # The convention I am using is: True = down and False = up
        counter = self.n - 1
        direction = True
        # This section of code creates the columns
        for x in xrange(0, self.size):
            if (x != counter) and direction:
                self.graph.add_edge(x, x + 1)
            elif (x == counter) and direction:
                self.graph.add_edge(x, x - (self.n - 1))
                self.looping_edges.append((x, (x - (self.n - 1))))
                counter += self.n
                direction = not direction
            elif ((counter - x) == (self.n - 1)) and not direction:
                self.graph.add_edge(x, counter)
                self.looping_edges.append((x, counter))
            elif x != counter and not direction:
                self.graph.add_edge(x, x - 1)
            elif x == counter and not direction:
                self.graph.add_edge(x, x - 1)
                counter += self.n
                direction = not direction

        # The counter variable is used to determine when to change the direction of the edge
        # The convention I am using is: True = right and False = left
        # c_val corresponds to the value right of x
        # c_num corresponds to the value in the rightmost column, on x's row
        # #####For n=3######## #
        #   x   c_val    c_num #
        #  (0) (4)   8  (12)   #
        #   1   5    9   13    #
        #   2   6    10  14    #
        #  (3)  7    11  15    #
        #  counter             #
        # #################### #
        counter = self.n - 1
        c_val = self.n
        c_num = self.size - self.n
        direction = True

        for x in xrange(0, self.size - self.n):
            if x != counter and direction:
                if x <= self.n - 1:
                    self.graph.add_edge(c_num, x)
                    self.looping_edges.append((c_num, x))
                    c_num += 1
                self.graph.add_edge(x, c_val)
                direction = not direction
                c_val += 1
            elif x != counter and not direction:
                if x <= self.n - 1:
                    self.graph.add_edge(x, c_num)
                    self.looping_edges.append((x, c_num))
                    c_num += 1
                self.graph.add_edge(c_val, x)
                direction = not direction
                c_val += 1
            elif x == counter and direction:
                if x <= self.n - 1:
                    self.graph.add_edge(c_num, x)
                    self.looping_edges.append((c_num, x))
                self.graph.add_edge(x, c_val)
                counter = c_val
                c_val += 1
            elif x == counter and not direction:
                if x <= self.n - 1:
                    self.graph.add_edge(x, c_num)
                    self.looping_edges.append((x, c_num))
                self.graph.add_edge(c_val, x)
                counter = c_val
                c_val += 1
                direction = not direction

    # START Graph Display Design ##########
    def create_node_positions(self):
        layout = {}
        counter = self.n - 1
        y_axis_counter = self.n
        x_axis_counter = 1

        for x in xrange(0, self.size):
            if x != counter:
                layout[x] = [x_axis_counter, y_axis_counter]
                y_axis_counter -= 1
            else:
                layout[x] = [x_axis_counter, y_axis_counter]
                y_axis_counter = self.n
                x_axis_counter += 1
                counter += self.n

        del counter, x_axis_counter, y_axis_counter
        return layout

    def create_edge_list(self):
        edge_list = [edge for edge in self.graph.edges() if edge not in self.looping_edges]
        # print edge_list
        return edge_list

    def edge_color_intensities(self):
        color_intensities = []
        for edge in self.create_edge_list():
            x_y = self.node_to_coordinates(self.tail(edge))
            x1_y1 = self.node_to_coordinates(self.head(edge))
            y = x_y[1]
            del x_y
            y1 = x1_y1[1]
            del x1_y1
            if y == y1:
                color_intensities.append(self.v_queues[self.head(edge)])
            else:
                color_intensities.append(self.h_queues[self.head(edge)])
        # print color_intensities
        return color_intensities

    def create_edge_labels(self):
        edge_labels = {}
        for edge in self.create_edge_list():
            x_y = self.node_to_coordinates(self.tail(edge))
            x1_y1 = self.node_to_coordinates(self.head(edge))
            y = x_y[1]
            del x_y
            y1 = x1_y1[1]
            del x1_y1
            if y == y1:
                edge_labels[edge] = (self.v_queues[self.head(edge)])
            else:
                edge_labels[edge] = (self.h_queues[self.head(edge)])
        return edge_labels

    # END Graph Display Design ############

    def predecessor(self, (u, v)):
        has_edge = self.graph.has_edge(u, v)
        assert has_edge, "The edge from %r" % u + " to %r" % v + " does not exist."
        del has_edge

        # Convert node u and v into (row, column) coordinates
        x_y = self.node_to_coordinates(u)
        x1_y1 = self.node_to_coordinates(v)
        x = x_y[0]
        y = x_y[1]
        del x_y
        x1 = x1_y1[0]
        y1 = x1_y1[1]
        del x1_y1

        # Vertical Movement
        if y == y1:
            if x < x1:
                if x == 0:
                    if x1 == (self.n - 1):
                        predecessor_edge = [((x + 1), y), (x, y)]
                    else:
                        predecessor_edge = [((self.n - 1), y), (x, y)]
                else:
                    predecessor_edge = [((x - 1), y), (x, y)]
            else:
                if x == (self.n - 1):
                    if x1 == 0:
                        predecessor_edge = [((x - 1), y), (x, y)]
                    else:
                        predecessor_edge = [(0, y), (x, y)]
                else:
                    predecessor_edge = [((x + 1), y), (x, y)]
        # Horizontal movement (y != y1 or x == x1)
        else:
            if y < y1:
                if y == 0:
                    if y1 == (self.n - 1):
                        predecessor_edge = [(x, (y + 1)), (x, y)]
                    else:
                        predecessor_edge = [(x, (self.n - 1)), (x, y)]
                else:
                    predecessor_edge = [(x, (y - 1)), (x, y)]
            else:
                if y == (self.n - 1):
                    if y1 == 0:
                        predecessor_edge = [(x, (y - 1)), (x, y)]
                    else:
                        predecessor_edge = [(x, 0), (x, y)]
                else:
                    predecessor_edge = [(x, (y + 1)), (x, y)]

        # Convert list of (row, column) coordinates back to (u, v) edge
        predecessor_edge = self.coordinates_to_edge(predecessor_edge[0], predecessor_edge[1])
        check = self.graph.has_edge(predecessor_edge[0], predecessor_edge[1])
        assert check, "A preceding edge was found but does not exist in the graph."
        del check
        return predecessor_edge

    def successor(self, (u, v)):
        has_edge = self.graph.has_edge(u, v)
        assert has_edge, "The edge from %r" % u + " to %r" % v + " does not exist."
        del has_edge

        # Convert node u and v into (row, column) coordinates
        x_y = self.node_to_coordinates(u)
        x1_y1 = self.node_to_coordinates(v)
        x = x_y[0]
        y = x_y[1]
        del x_y
        x1 = x1_y1[0]
        y1 = x1_y1[1]
        del x1_y1

        # Vertical movement
        if y == y1:
            if x < x1:
                if x1 == (self.n - 1) and (y % 2) == 0:
                    successor_edge = [(x1, y), (0, y)]
                elif x1 == (self.n - 1) and (y % 2) != 0:
                    successor_edge = [(x1, y), ((x1 - 1), y)]
                else:
                    successor_edge = [(x1, y), ((x1 + 1), y)]
            else:
                if x1 == 0 and (y % 2) == 0:
                    successor_edge = [(x1, y), ((x1 + 1), y)]
                elif x1 == 0 and (y % 2) != 0:
                    successor_edge = [(x1, y), ((self.n - 1), y)]
                else:
                    successor_edge = [(x1, y), ((x1 - 1), y)]
        # Horizontal movement (x == x1 or y != y1)
        else:
            if y < y1:
                if y1 == (self.n - 1) and (x % 2) == 0:
                    successor_edge = [(x, y1), (x, 0)]
                elif y1 == (self.n - 1) and (x % 2) != 0:
                    successor_edge = [(x, y1), (x, (y1 - 1))]
                else:
                    successor_edge = [(x, y1), (x, (y1 + 1))]
            else:
                if y1 == 0 and (x % 2) == 0:
                    successor_edge = [(x, y1), (x, (y1 + 1))]
                elif y1 == 0 and (x % 2) != 0:
                    successor_edge = [(x, y1), (x, (self.n - 1))]
                else:
                    successor_edge = [(x, y1), (x, (y1 - 1))]

        # Convert list of (row, column) coordinates back to (u, v) edge
        successor_edge = self.coordinates_to_edge(successor_edge[0], successor_edge[1])
        check = self.graph.has_edge(successor_edge[0], successor_edge[1])
        assert check, "A succeeding edge was found but does not exist in the graph."
        del check
        return successor_edge

    def conflict(self, (u, v)):
        has_edge = self.graph.has_edge(u, v)
        assert has_edge, "The edge from %r" % u + " to %r" % v + " does not exist."
        del has_edge

        # Convert node u and v into (row, column) coordinates
        x_y = self.node_to_coordinates(u)
        x1_y1 = self.node_to_coordinates(v)
        x = x_y[0]
        y = x_y[1]
        del x_y
        x1 = x1_y1[0]
        y1 = x1_y1[1]
        del x1_y1

        # Vertical movement
        if y == y1:
            if x < x1:
                if y == 0:
                    if (x1 % 2) == 0:
                        conflicting_edge = [(x1, (self.n - 1)), (x1, y)]
                    else:
                        conflicting_edge = [(x1, (y + 1)), (x1, y)]
                elif y == (self.n - 1):
                    if (x1 % 2) == 0:
                        conflicting_edge = [(x1, (y - 1)), (x1, y)]
                    else:
                        conflicting_edge = [(x1, 0), (x1, y)]
                else:
                    if (x1 % 2) == 0:
                        conflicting_edge = [(x1, (y - 1)), (x1, y)]
                    else:
                        conflicting_edge = [(x1, (y + 1)), (x1, y)]
            else:
                if y == 0:
                    conflicting_edge = [(x1, (self.n - 1)), (x1, y)]
                elif y == (self.n - 1):
                    if (x1 % 2) == 0:
                        conflicting_edge = [(x1, (y - 1)), (x1, y)]
                    else:
                        conflicting_edge = [(x1, 0), (x1, y)]
                else:
                    if (x1 % 2) == 0:
                        conflicting_edge = [(x1, (y - 1)), (x1, y)]
                    else:
                        conflicting_edge = [(x1, (y + 1)), (x1, y)]

        # Horizontal movement
        else:
            if y < y1:
                if x == 0:
                    if (y1 % 2) == 0:
                        conflicting_edge = [((self.n - 1), y1), (x, y1)]
                    else:
                        conflicting_edge = [((x + 1), y1), (x, y1)]
                elif x == (self.n - 1):
                    if (y1 % 2) == 0:
                        conflicting_edge = [((x - 1), y1), (x, y1)]
                    else:
                        conflicting_edge = [(0, y1), (x, y1)]
                else:
                    if (y1 % 2) == 0:
                        conflicting_edge = [((x - 1), y1), (x, y1)]
                    else:
                        conflicting_edge = [((x + 1), y1), (x, y1)]
            else:
                if x == 0:
                    conflicting_edge = [((self.n - 1), y1), (x, y1)]
                elif x == (self.n - 1):
                    if (y1 % 2) == 0:
                        conflicting_edge = [((x - 1), y1), (x, y1)]
                    else:
                        conflicting_edge = [(0, y1), (x, y1)]
                else:
                    if (y1 % 2) == 0:
                        conflicting_edge = [((x - 1), y1), (x, y1)]
                    else:
                        conflicting_edge = [((x + 1), y1), (x, y1)]

        # Convert list of (row, column) coordinates back to (u, v) edge
        conflicting_edge = self.coordinates_to_edge(conflicting_edge[0], conflicting_edge[1])
        check = self.graph.has_edge(conflicting_edge[0], conflicting_edge[1])
        assert check, "A conflicting edge was found but does not exist in the graph."
        del check
        return conflicting_edge

    def backward_conflict(self, (u, v)):
        return self.successor(self.conflict(self.predecessor((u, v))))

    def n_plus(self, (u, v)):
        forward_neighbors = [self.successor((u, v)), self.conflict((u, v))]
        return forward_neighbors

    def n_minus(self, (u, v)):
        backward_neighbors = [self.predecessor((u, v)), self.backward_conflict((u, v))]
        return backward_neighbors

    def f_plus(self, (u, v), x_value):
        has_edge = self.graph.has_edge(u, v)
        assert has_edge, "The edge from %r" % u + " to %r" % v + " does not exist."
        del has_edge

        assert abs(self.shifts[self.head((u, v))] + x_value) <= (self.psi / 2), \
            "x = %r is not less than or equal to psi/2." % abs(self.shifts[self.head((u, v))] + x_value)

        # Convert node u and v into (row, column) coordinates
        x_y = self.node_to_coordinates(u)
        x1_y1 = self.node_to_coordinates(v)
        x = x_y[0]
        y = x_y[1]
        del x_y
        x1 = x1_y1[0]
        y1 = x1_y1[1]
        del x1_y1

        # Horizontal movement
        if x == x1:
            self.shifts[self.head((u, v))] += x_value
        # Vertical movement
        else:
            self.shifts[self.head((u, v))] -= x_value

        assert (abs(self.shifts[self.head((u, v))]) <= (self.psi / 2)), "The updated |shift| = %r " \
                                                                        % abs(self.shifts[self.head((u, v))]) + \
                                                                        "is not less than or equal to psi / 2."
        del x, y, x1, y1

    def f_minus(self, (u, v), x_value):
        has_edge = self.graph.has_edge(u, v)
        assert has_edge, "The edge from %r" % u + " to %r" % v + " does not exist."
        del has_edge

        assert abs(self.shifts[self.tail((u, v))] + x_value) <= (self.psi / 2), \
            "x = %r is not less than or equal to |shift|." % x_value

        # Convert node u and v into (row, column) coordinates
        x_y = self.node_to_coordinates(u)
        x1_y1 = self.node_to_coordinates(v)
        x = x_y[0]
        y = x_y[1]
        del x_y
        x1 = x1_y1[0]
        y1 = x1_y1[1]
        del x1_y1

        # Horizontal movement
        if x == x1:
            self.shifts[self.tail((u, v))] -= x_value
        # Vertical movement
        else:
            self.shifts[self.tail((u, v))] += x_value

        assert (abs(self.shifts[self.tail((u, v))]) <= (self.psi / 2)), "The updated |shift| = %r " \
                                                                        % abs(self.shifts[self.tail((u, v))]) + \
                                                                        "is not less than or equal to psi / 2."
        del x, y, x1, y1

    def w(self, (u, v)):
        has_edge = self.graph.has_edge(u, v)
        assert has_edge, "The edge from %r" % u + " to %r" % v + " does not exist."
        del has_edge

        # Convert node u and v into (row, column) coordinates
        x_y = self.node_to_coordinates(u)
        x1_y1 = self.node_to_coordinates(v)
        x = x_y[0]
        del x_y
        x1 = x1_y1[0]
        del x1_y1

        direction = False
        # Horizontal movement
        if x == x1:
            direction = True
            w = int(self.h_queues[self.head((u, v))] +
                    min(((self.psi / 2) + self.shifts[self.head(self.predecessor((u, v)))]),
                        self.h_queues[self.head(self.predecessor((u, v)))]) -
                    min(((self.psi / 2) + self.shifts[self.head((u, v))]), self.h_queues[self.head((u, v))]))
        # Vertical movement
        else:
            w = int(self.v_queues[self.head((u, v))] +
                    min(((self.psi / 2) - self.shifts[self.head(self.predecessor((u, v)))]),
                        self.v_queues[self.head(self.predecessor((u, v)))]) -
                    min(((self.psi / 2) - self.shifts[self.head((u, v))]), self.v_queues[self.head((u, v))]))
        return w, direction

    def minimizing(self):
        cycle = False
        print "############\n" \
              "#MINIMIZING#\n" \
              "############"
        iteration = 1
        count_deploy = 0
        while not cycle:
            print "Minimizing Iteration %r" % iteration
            e = self.find_largest_queue()
            print "The edge with the largest queue is: %r" % str(e)
            shifts_backup = list(self.shifts)
            phi = self.w(e)[0]
            print "phi is: %r" % phi
            (state, saturated, pi) = self.flooding(e, "+", (phi - 1))
            if not state:
                if saturated:
                    print "Saturated"
                    self.shifts = list(shifts_backup)
                    self.deploy()
                    count_deploy += 1
                    if count_deploy >= 2:
                        cycle = True
                else:
                    print "Not state 1"
                    self.shifts = list(shifts_backup)
                    (state, saturated, pi) = self.flooding(e, "-", (phi - 1))
                    if not state:
                        print "Not state 2. There is a cycle"
                        cycle = True
                        self.shifts = list(shifts_backup)
                        # print "Final total weights: %r" % self.check_weights()
            else:
                print "Iteration %r " % iteration + "completed successfully.\n"
                count_deploy = 0
            iteration += 1
        self.deploy()
        # print "Final2 total weights: %r" % self.check_weights()
        print "############\n" \
              "#END########\n" \
              "#MINIMIZING#\n" \
              "############\n"
        return phi, pi

    def find_largest_queue(self):
        phi = 0
        for edge in self.graph.edges():
            if phi < self.w(edge)[0]:
                phi = self.w(edge)[0]
                e = edge
        return e

    def flooding(self, root, direction, capacity):
        print "##########\n" \
              "#FLOODING#\n" \
              "##########"
        iteration = 1
        print "Direction is " + direction
        Q = []
        print "pi[root] is now %r = root" % str(root)
        pi = {root: root}

        if self.w(root)[0] > capacity:
            print "W of root = %r" % self.w(root)[0] + " is > %r = capacity" % capacity
            print "Adding root = " + str(root) + " to the stack"
            Q.append(root)
        else:
            print "W of root = %r" % self.w(root)[0] + " is < %r = capacity" % capacity

        while len(Q) is not 0:

            print "\nFlooding Iteration %r" % iteration
            print "Total weight: %r" % self.check_weights()
            e = Q.pop()
            print "Popped %r from the stack" % str(e)
            print "Weight of e = %r" % self.w(e)[0]
            print "N- of e = " + str(e) + " is " + str(self.n_minus(e))
            print "N+ of e = " + str(e) + " is " + str(self.n_plus(e))

            if pi[e] in self.n_minus(e) or direction == "+":
                print "Processing edge " + str(e) + ". Either pi[e] = %r in N- of e or direction == +" \
                                                    % str(pi[e])
                print "Shift of edge(head) prior to f_plus = %r" % self.shifts[self.head(e)]

                if not self.valid_shift_plus(e, (self.w(e)[0] - capacity)):
                    return False, True, pi

                self.f_plus(e, (self.w(e)[0] - capacity))
                print "Shift of edge(head) after f_plus = %r" % self.shifts[self.head(e)]

                if (self.successor(e) in pi.iterkeys() and (self.w(self.successor(e))[0] > capacity)) \
                        or (self.conflict(e) in pi.iterkeys() and (self.w(self.conflict(e))[0] > capacity)) \
                        or abs(self.shifts[self.head(e)]) > (self.psi / 2):
                    print "pi = " + str(pi)
                    print "shifts = %r" % str(self.shifts)
                    print "Exit Flooding Case 1"
                    print "##########\n" \
                          "#END######\n" \
                          "#FLOODING#\n" \
                          "##########\n"
                    return False, False, pi

                if self.w(self.successor(e))[0] > capacity:
                    print "Adding the successor of e: " + str(self.successor(e)) + " to the stack"
                    Q.append(self.successor(e))
                    print "The parent of " + str(self.successor(e)) + " is " + str(e)
                    pi[self.successor(e)] = e
                else:
                    print "W of successor of e = %r" % self.w(self.successor(e))[0] + " < %r = capacity" % capacity

                if self.w(self.conflict(e))[0] > capacity:
                    print "W of conflict of e = %r" % self.w(self.conflict(e))[0] + " > %r = capacity" % capacity
                    print "Adding the conflict of e: " + str(self.conflict(e)) + " to the stack"
                    Q.append(self.conflict(e))
                    print "The parent of " + str(self.conflict(e)) + " is " + str(e)
                    pi[self.conflict(e)] = e
                else:
                    print "W of conflict of e = %r" % self.w(self.conflict(e))[0] + " < %r = capacity" % capacity

            elif pi[e] in self.n_plus(e) or direction == "-":
                print "Processing edge " + str(e) + ". Either pi[e] = %r in N+ of e or direction == -" \
                                                    % str(pi[e])
                print "Shift of edge(tail) prior to f_minus = %r" % self.shifts[self.tail(e)]

                if not self.valid_shift_minus(self.predecessor(e), (self.w(e)[0] - capacity)):
                    return False, True, pi

                self.f_minus(e, (self.w(e)[0] - capacity))
                print "Shift of edge(tail) after f_minus = %r" % self.shifts[self.tail(e)]

                if (self.predecessor(e) in pi.keys() and (self.w(self.predecessor(e))[0] > capacity)) \
                        or self.backward_conflict(e) in pi.keys() and \
                                (self.w(self.backward_conflict(e))[0] > capacity) \
                        or abs(self.shifts[self.head(self.predecessor(e))]) > (self.psi / 2):
                    print "pi = " + str(pi)
                    print "shifts = %r" % str(self.shifts)
                    print "Exit Case 2"
                    print "##########\n" \
                          "#END######\n" \
                          "#FLOODING#\n" \
                          "##########\n"
                    return False, False, pi

                if self.w(self.predecessor(e))[0] > capacity:
                    print "Adding the predecessor of e: " + str(self.predecessor(e)) + " to the stack"
                    Q.append(self.predecessor(e))
                    print "The parent of " + str(self.predecessor(e)) + " is " + str(e)
                    pi[self.predecessor(e)] = e
                else:
                    print "W of predecessor of e = %r" % self.w(self.predecessor(e))[0] + " < %r = capacity" % capacity

                if self.w(self.backward_conflict(e))[0] > capacity:
                    print "Adding the bconf of e: " + str(self.backward_conflict(e)) + " to the stack"
                    Q.append(self.backward_conflict(e))
                    print "The parent of " + str(self.backward_conflict(e)) + " is " + str(e)
                    pi[self.backward_conflict(e)] = e
                else:
                    print "W of bconf of e = %r" % self.w(self.backward_conflict(e))[0] + " < %r = capacity" % capacity

            direction = "null"
            print "pi = " + str(pi)
            print "The length of Q is: %r" % len(Q)
            print "shifts = %r" % str(self.shifts)
            iteration += 1
        print "##########\n" \
              "#END######\n" \
              "#FLOODING#\n" \
              "##########\n"
        return True, False, pi

    def valid_shift(self, (u, v), x_value):
        has_edge = self.graph.has_edge(u, v)
        assert has_edge, "The edge from %r" % u + " to %r" % v + " does not exist."
        del has_edge

        x_y = self.node_to_coordinates(u)
        x1_y1 = self.node_to_coordinates(v)
        x = x_y[0]
        y = x_y[1]
        del x_y
        x1 = x1_y1[0]
        y1 = x1_y1[1]
        del x1_y1

        # Horizontal movement
        if x == x1:
            print "Horizontal edge %r" % str((u, v)) + " abs(self.shifts[self.head((u,v))] + x_value) = %r" \
                                                       % abs(self.shifts[self.head((u, v))] + x_value)
            print "psi/2 = %r" % (self.psi / 2)
            return abs(self.shifts[self.head((u, v))] + x_value) <= (self.psi / 2)
        else:
            print "Vertical edge %r" % str((u, v)) + " abs(self.shifts[self.head((u,v))] - x_value) = %r" \
                                                     % abs(self.shifts[self.head((u, v))] - x_value)
            print "psi/2 = %r" % (self.psi / 2)
            return abs(self.shifts[self.head((u, v))] - x_value) <= (self.psi / 2)

    def valid_shift_plus(self, (u, v), x_value):

        # Convert node u and v into (row, column) coordinates
        x_y = self.node_to_coordinates(u)
        x1_y1 = self.node_to_coordinates(v)
        x = x_y[0]
        y = x_y[1]
        del x_y
        x1 = x1_y1[0]
        y1 = x1_y1[1]
        del x1_y1

        # Horizontal movement
        if x == x1:
            return abs(self.shifts[self.head((u, v))] + x_value) < (self.psi / 2)
        # Vertical movement
        else:
            return abs(self.shifts[self.head((u, v))] - x_value) < (self.psi / 2)

    def valid_shift_minus(self, (u, v), x_value):

        # Convert node u and v into (row, column) coordinates
        x_y = self.node_to_coordinates(u)
        x1_y1 = self.node_to_coordinates(v)
        x = x_y[0]
        y = x_y[1]
        del x_y
        x1 = x1_y1[0]
        y1 = x1_y1[1]
        del x1_y1

        # Horizontal movement
        if x == x1:
            return abs(self.shifts[self.tail((u, v))] - x_value) < (self.psi / 2)
        # Vertical movement
        else:
            return abs(self.shifts[self.tail((u, v))] + x_value) < (self.psi / 2)

    def deploy(self):
        h_queues = list(self.h_queues)
        v_queues = list(self.v_queues)
        total_weight = 0
        for edge in self.graph.edges():
            w = self.w(edge)
            total_weight += w[0]
            if w[1]:
                h_queues[self.head(edge)] = w[0]
            else:
                v_queues[self.head(edge)] = w[0]
        self.h_queues = list(h_queues)
        self.v_queues = list(v_queues)
        del h_queues, v_queues
        print "The total weight from deploy: %r" % total_weight

    def check_weights(self):
        total_weight = 0
        for edge in self.graph.edges():
            total_weight += self.w(edge)[0]
        return total_weight

    def output(self, node):
        successors = self.graph.successors(node)
        edges = []
        for x in successors:
            edges.append((node, x))
        del successors
        return edges

    def input(self, node):
        predecessors = self.graph.predecessors(node)
        edges = []
        for x in predecessors:
            edges.append((x, node))
        del predecessors
        return edges

    def index(self, i, j):
        x = (j * self.n) + i
        return x

    def row_number(self, index):
        i = index % self.n
        return i

    def column_number(self, index):
        j = int(math.floor(index / self.n))
        return j

    def node_to_coordinates(self, node):
        i = node % self.n
        j = int(math.floor(node / self.n))
        coordinate = (i, j)
        return coordinate

    def coordinates_to_edge(self, (x, y), (x1, y1)):
        u = self.index(x, y)
        v = self.index(x1, y1)
        has_edge = self.graph.has_edge(u, v)
        assert has_edge, "The edge from %r" % u + " to %r" % v + " does not exist."
        del has_edge
        edge = (u, v)
        return edge

    def head(self, (u, v)):
        has_edge = self.graph.has_edge(u, v)
        assert has_edge, "The edge from %r" % u + " to %r" % v + " does not exist."
        del has_edge
        return v

    def tail(self, (u, v)):
        has_edge = self.graph.has_edge(u, v)
        assert has_edge, "The edge from %r" % u + " to %r" % v + " does not exist."
        del has_edge
        return u

    # END CLASS ############################################
