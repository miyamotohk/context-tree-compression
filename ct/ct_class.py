################################################
# Context-tree functions
################################################

from math import floor, ceil, log2

### Tree nodes
class Node:
	"""
	Main class for the nodes of the context-tree.
	"""
	def __init__(self, m, context=[]):
		self.m         = m                                       # Alphabet size
		self.context   = context                                 # Context
		self.d         = len(context)                            # Tree depth
		self.label     = ''
		self.label     += ''.join(str(e) for e in context[::-1]) # Node string label
		self.count     = [0] * m                                 # Count for each symbol
		self.children  = [None] * m                              # Children nodes
		self.log2Pe    = 0                                       # log2 of estimated probability Pe
		self.log2Pw    = 0                                       # log2 of weighted probability Pw
		self.log2Pm    = 0                                       # log2 of maximising probability Pm
		self.isToPrune = False                                   # Flag indicating whether to prune or not

	def isLeaf(self):
		return all(child is None for child in self.children)

	def countLeaves(self):
		return sum(1 for n in buildIter(self) if n.isLeaf())

	def countLeavesAtDepth(self, D):
		return sum(1 for n in buildIter(self) if n.d==D)

	def prune(self):
		self.children = [None] * self.m

	def getlog2Pe(self, alpha=2):
		Ms = sum(self.count)
		if Ms == 0:
			return 0

		num = log2gamma(self.m/alpha)
		for i in range(self.m):
			num += log2gamma((1/alpha) + self.count[i])
		
		den = self.m*log2gamma(1/alpha) + log2gamma((self.m/alpha) + Ms)
		return (num-den)

	def getlog2Pw(self, beta=0.5):
		if self.isLeaf():
			return self.log2Pe
		else:
			log2p1 = log2(beta) + self.log2Pe
			log2p2 = log2(1-beta) + sum(c.log2Pw for c in self.children if c is not None)
			if log2p2 > log2p1:
				return log2p2 + log2((2**(log2p1-log2p2))+1)
			else:
				return log2p1 + log2((2**(log2p2-log2p1))+1)

	def getlog2Pm(self, beta=0.5, pruneFlag=True):
		if self.isLeaf():
			return self.log2Pe
		else:
			if log2(beta) + self.log2Pe >= log2(1-beta) + sum(c.log2Pm for c in self.children if c is not None):
				self.isToPrune = True
				if pruneFlag:
					self.prune()
				return log2(beta) + self.log2Pe
			else:
				self.isToPrune = False
				return log2(1-beta) + sum(c.log2Pm for c in self.children if c is not None)

	def getCTWProbs(self, D, beta=0.5, alpha=2):
		for child in self.children:
			if child != None:
				child.getCTWProbs(D, beta, alpha)
		self.log2Pe = self.getlog2Pe(alpha)
		self.log2Pw = self.getlog2Pw(beta)

	def getCTMProbs(self, D, beta=0.5, alpha=2, pruneFlag=True):
		for child in self.children:
			if child != None:
				child.getCTMProbs(D, beta, alpha, pruneFlag)
		self.log2Pe = self.getlog2Pe(alpha)
		self.log2Pm = self.getlog2Pm(beta, pruneFlag)

### Auxiliary functions
def log2gamma(x):
	res = 0
	x -= 1
	while x > 0:
		res += log2(x)
		x -= 1
	return res

def product(iter):
	result = 1
	for elem in iter:
		result *= elem
	return result