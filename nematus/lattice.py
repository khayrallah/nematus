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

    def numWords(self):
        return len(self.label.split('_'))

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


class BestItem:
    '''Data structure for recording best item in graph'''

    def __init__(self, score=-999999.0, state=None, arc=None, pathLength = 0):
        self.score = score
        self.state = state
        self.arc = arc
        self.pathLength = pathLength

    def normalizedScore(self):
        if self.pathLength > 0:
            return self.score / float(self.pathLength)
        else:
            return self.score


class Graph:
    def __str__(self):
        return `self.sentno`

    def __init__(self, sentno):
        self.sentno = sentno
        self.root = None
        self.nodelist = []
        self.arccount = 0
        self.finalstate = -1
        self.nodes = {}
        self.root = self.node(0)

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

    def walk(self, scorer = None, normalize = True, verbose = False):

        if scorer is None:
            scorer = self

        # for each node, the best state, the word that produced it, and the cumulative score
        bestitems = {}
        for node in self.nodelist:
            if verbose: print 'processing {} with {} incoming arcs'.format(node, len(node.getIncomingArcs()))
            
            best = BestItem()
            for arc in node.getIncomingArcs():
                prevBest = bestitems.get(arc.tail, BestItem(score = 0.))
                oldscore, state = prevBest.score, prevBest.state
                
                newstate, transitioncost = scorer.score(state, arc)
                score = oldscore + transitioncost
                pathLength = prevBest.pathLength + arc.numWords()

                if verbose: print '  {} -> {}'.format(arc, score)
                if normalize:
                    normalizedScore = score / float(pathLength)
                    if normalizedScore > best.normalizedScore():
                        if verbose: print '  new best ({} > {})'.format(normalizedScore, best.score)
                        best = BestItem(score, newstate, arc, pathLength)

                else:
                    if score > best.score:
                        if verbose: print '  new best ({} > {})'.format(score, best.score)
                        best = BestItem(score, newstate, arc, pathLength)

            if best.arc is not None:
                bestitems[node] = best
                if verbose: print 'best -> {} is {} ({})'.format(best.arc.head, best.arc.label, best.arc.score)

        # Now follow the backpointers to construct the final sentence
        finalnode = self.node(self.finalstate)
        seq = []
        while node.id != 0:
            arc = bestitems.get(node).arc
            seq.insert(0, arc.label)
            node = arc.tail

        print self.sentno, bestitems.get(finalnode).score, ' '.join(seq).replace('<eps>','').replace('_', ' ').strip()


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
