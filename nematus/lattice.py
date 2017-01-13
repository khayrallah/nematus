"""
Lattice class. This represents a lattice. It supports loading from an OpenFST file.
"""

class Arc:
    head = None
    tail = None
    label = None
    score = None

    def __str__(self):
        return 'ARC[{} -> {} -> {}, {}]'.format(self.tail, self.label, self.head, self.score)

    def __init__(self, tail, label, head, score):
        self.tail = tail
        self.label = label
        self.head = head
        self.score = float(score)

class Node:
    def __init__(self, id = 0):
        self.id = id
        self.outarcs = []
        self.inarcs = []

    def __str__(self):
        return 'NODE[{}]'.format(self.id)

    def addOutgoingArc(self, arc):
        self.outarcs.append(arc)

    def addIncomingArc(self, arc):
        self.inarcs.append(arc)

    def getIncomingArcs(self):
        return self.inarcs

class Graph:
    sentno = -1
    root = None
    nodes = {}
    nodelist = []
    arccount = 0
    finalstate = -1

    def __str__(self):
        return `self.sentno`

    def __init__(self, sentno):
        self.sentno = sentno
        root = self.node(0)

    def id(self):
        return self.sentno

    def numnodes(self):
        return len(self.nodes)

    def numarcs(self):
        return self.arccount

    def addArc(self, tail, label, head, score):
        arc = Arc(self.node(tail), label, self.node(head), score)
        self.node(tail).addOutgoingArc(arc)
        self.node(head).addIncomingArc(arc)
        self.arccount += 1

    def node(self, id):
        id = int(id)
        if not self.nodes.has_key(id):
            node = Node(id)
            self.nodes[id] = node
            self.nodelist.append(node)

        return self.nodes[id]

    def score(self, state, arc):
        """The default scoring option. Returns the score read in on the arc, ignoring the old state and not returning a new one."""
        return None, arc.score

    def walk(self, scorer = None):
        if scorer is None:
            scorer = self

        # for each node, the best state, the word that produced it, and the cumulative score
        states = {}
        for node in self.nodelist:
#            print 'processing {} with {} incoming arcs'.format(node, len(node.getIncomingArcs()))
            best = (-999999, None, None)
            for arc in node.getIncomingArcs():
                oldscore, state, word = states.get(arc.tail, (0, None, ""))

                newstate, transitioncost = scorer.score(state, arc)
                score = oldscore + transitioncost

#                print '  {} -> {}'.format(arc, score)
                if score > best[0]:
#                    print '  new best ({} > {})'.format(score, best[0])
                    best = (score, newstate, arc)
                # else:
                #     print '  old is better ({} < {})'.format(score, best[0])

            if best[2] is not None:
                states[node] = best
                arc = best[2]
                # print 'best -> {} is {} ({})'.format(arc.head, arc.label, arc.score)

        node = self.node(self.finalstate)
        # print 'best modelscore =', states[node][0]
        seq = []
        while node.id != 0:
            score, state, arc = states.get(node)
            seq.insert(0, arc.label)
            node = arc.tail

        print states[self.node(self.finalstate)][0], ' '.join(seq).replace('<eps>','').replace('_', ' ').strip()


def read_graph(search_graph_file, sentno = 0):
    """
    Reads an OpenFST file and constructs a walkable graph.
    """

    graph = Graph(sentno)
    for line in search_graph_file:
        try:
            tail, head, source, target = line.rstrip().split(' ', 3)
        except ValueError:
            graph.finalstate = int(line.rstrip())

        if ' ' in target:
            target, score = target.split(' ')
        else:
            score = 0.0

        tail = int(tail)
        head = int(head)
        graph.addArc(tail, target, head, -float(score))

#    print "graph[->{}] {} has {} nodes and {} arcs".format(graph.finalstate, graph.id(), graph.numnodes(), graph.numarcs())
    return graph
