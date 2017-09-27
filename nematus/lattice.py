"""
Lattice class. This represents a lattice. It supports loading from an OpenFST file.
"""

from heapq import *

WORD_DELIM='|'

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

    def __hash__(self):
        return hash(self.tail, self.label, self.head)

    def __eq__(self, other):
        return (self.tail, self.label, self.head) == (other.tail, other.label, other.head)

    def words(self):
        return self.label.split(WORD_DELIM)

    def numWords(self):
        return len(self.words())

class Node:
    def __init__(self, id = 0):
        self.id = id
        self.outarcs = []
        self.inarcs = []

    def __str__(self):
        return 'NODE[{}]'.format(self.id)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id

    def addOutgoingArc(self, arc):
        self.outarcs.append(arc)

    def addIncomingArc(self, arc):
        self.inarcs.append(arc)

    def getIncomingArcs(self):
        return self.inarcs

    def getOutgoingArcs(self):
        return self.outarcs

class BestItem:
    '''Data structure for recording best item in graph'''

    def __init__(self, score=-999999.0, state=None, arc=None, pathLength=0, prevBest=None):
        self.score = score
        self.state = state
        self.arc = arc
        self.pathLength = pathLength
        self.prev = prevBest

    def __str__(self):
        return 'ITEM[{}, {}, {}]'.format(self.arc, self.pathLength, self.score)

    def __lt__(self, other):
        """Actually returns greater than."""
        return self.score > other.score

    # def __hash__(self):
    #     return hash((self.state, self.arc))

    # def __eq__(self, other):
    #     return (self.state, self.arc) == (other.state, other.arc)

    def normalizedScore(self):
        if self.pathLength > 0:
            return self.score / float(self.pathLength)
        else:
            return self.score

class Graph:
    def __str__(self):
        return `self.sentno`

    def __init__(self, sentno, search_graph_file = None):
        self.sentno = sentno
        self.root = None
        self.nodelist = []
        self.arccount = 0
        self.finalstate = -1
        self.nodes = {}
        self.root = self.node(0)

        if search_graph_file is not None:
            self.read_graph(search_graph_file)

    def read_graph(self, search_graph_file):
        """
        Reads an OpenFST file and constructs a walkable graph.
        """

        for line in search_graph_file:
            try:
                tail, head, source, target = line.rstrip().split(None, 3)
            except ValueError:
                self.finalstate = int(line.rstrip())

            if len(target.split()) > 1:
                target, score = target.split()
            else:
                score = 0.0

            tail = int(tail)
            head = int(head)
            self.addArc(tail, target, head, -float(score))

        self.nodelist = sorted(self.nodelist, cmp=lambda x, y: cmp(x.id, y.id))

    #    print "graph[->{}] {} has {} nodes and {} arcs".format(graph.finalstate, graph.id(), graph.numnodes(), graph.numarcs())


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

    def extractBest(self, origitem, verbose = False):
        words = []
        item = origitem
        while item.arc is not None:
            words.insert(0, item.arc.label)
            if verbose:
                print "BESTARC: {}".format(item.arc)
            item = item.prev

        return self.sentno, origitem.score, ' '.join(words).replace('<eps>','').replace(WORD_DELIM, ' ').strip()

    def beam_search(self, scorer = None, verbose = True, beam = 12):
        '''
        Performs beam search over the search graph. Paths are grouped by how many target words they represent, 
        and cube pruning is applied to each.
        '''
        if scorer is None:
            scorer = self

        heaps = []
        heaps.append([ BestItem(score = 0.) ])

        # Any time we encounter a final state, it gets added here
        finalitems = []

        # iterate over the stacks
        heapno = 0
        while heapno < len(heaps):
            heap = heaps[heapno]
            if verbose: print "STACK {} WITH {} ITEMS".format(heapno, len(heap))

            # Pop items off the stack, extending hypotheses and adding them into later stacks
            beam_i = 0
            while beam_i < beam and len(heap) > 0:
                # Get the next item
                item = heappop(heap)
                if item.arc is not None:
                    node = item.arc.head
                else:
                    node = self.nodelist[0]

                if verbose: print "BEAM: POP {} -> {} ({} arcs)".format(beam_i, item, len(node.getOutgoingArcs()))

                # Add its extensions
                for arc in node.getOutgoingArcs():
                    newstate, transitioncost = scorer.score(item.state, arc)
                    score = item.score + transitioncost
                    pathLength = item.pathLength + arc.numWords()

                    nextstackno = heapno + arc.numWords()
                    while len(heaps) <= nextstackno:
                        heaps.append([])

                    nextitem = BestItem(score, newstate, arc, pathLength, item)

                    if len(arc.head.getOutgoingArcs()) == 0:
                        heappush(finalitems, nextitem)
                    else:
                        heappush(heaps[nextstackno], nextitem)

                    if verbose: print '  {} -> {}'.format(arc, score)

                beam_i += 1

            heapno += 1

        finalitem = heappop(finalitems)
        result = self.extractBest(finalitem, verbose)
        return result


    def walk(self, scorer = None, normalize = False, verbose = False):

        if scorer is None:
            scorer = self

        # for each node, the best state, the word that produced it, and the cumulative score
        bestitems = {}
        for node in self.nodelist:
            if verbose: print 'processing {} with {} incoming arcs'.format(node, len(node.getIncomingArcs()))
            
            best = None
            for arc in node.getIncomingArcs():
                prevBest = bestitems.get(arc.tail, BestItem(score = 0.))
                oldscore, state = prevBest.score, prevBest.state
                
                newstate, transitioncost = scorer.score(state, arc)
                score = oldscore + transitioncost
                pathLength = prevBest.pathLength + arc.numWords()

                if verbose: print '  {} -> {}'.format(arc, score)
                if normalize:
                    normalizedScore = score / float(pathLength)
                    if best is None or normalizedScore > best.normalizedScore():
                        if verbose: 
                            bestscore = best.normalizedScore() if best is not None else 0.
                            print '  new best ({} > {})'.format(normalizedScore, bestscore)
                        best = BestItem(score, newstate, arc, pathLength, prevBest)

                else:
                    if best is None or score > best.score:
                        if verbose: 
                            bestscore = best.score if best is not None else 0.
                            print '  new best ({} > {})'.format(score, bestscore)
                        best = BestItem(score, newstate, arc, pathLength, prevBest)

            if best is not None:
                bestitems[node] = best
                if verbose: print 'best -> {} is {} ({})'.format(best.arc.head, best.arc.label, best.arc.score)

        # Now follow the backpointers to construct the final sentence
        finalnode = self.node(self.finalstate)
        finalitem = bestitems[finalnode]
        result = self.extractBest(finalitem, verbose)
        return result
