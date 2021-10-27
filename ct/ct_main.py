################################################
# Context-tree functions
################################################

from math import floor, ceil, log2

from ct_class import Node
from ct_draw  import drawTree

def buildTree(data, m, D, type, beta=0.5, alpha=2, pruneFlag=True):
	"""
	Build context tree for a given sequence.
	Input:
		data:      sequence of symbols
		m:         alphabet size
		D:         tree maximum depth
		beta:      weighting coefficient
		alpha:     Dirichlet coefficient
		type:      'CTW' or 'CTM'
		pruneFlag: flag indicating whether to prune tree or not
	"""
	root = Node(m)

	# Create full empty tree
	createEmptyDescendants(root,[],D)

	for tau in range(1, len(data)-D+1):
		start   = tau - 1
		context = data[start:start+D]
		nextSym = data[start+D]
		increaseCount(root, context, nextSym)

	#completeTree(root)

	if type == 'CTW':
		root.getCTWProbs(D, beta, alpha)
	elif type == 'CTM':
		root.getCTMProbs(D, beta, alpha, pruneFlag)
	else:
		print("ERROR: Type not recognised. Try 'CTW or 'CTM'.")
		return -1

	return root

def createEmptyDescendants(node, context, D):
	"""
	Create descendants with zeroed counts.
	Inputs:
		node:    tree node
		context: current context
		D:       tree maximum depth
	"""
	m = node.m

	if node.d < D:
		for i in range(m):
			newContext = [i]+context
			node.children[i] = Node(m, newContext)
			createEmptyDescendants(node.children[i], newContext, D)

def increaseCount(root, context, nextSym):
	"""
	Increase counts and nodes if necessary.
	Inputs:
		root:    root node of the tree
		context: context of next symbol
		nextSym: next symbol
	"""
	i = j = 0

	m       = root.m
	node    = root
	context = context[::-1]  # Invert context

	# Navigate tree
	for j in range(len(context)):
		# If the corresponding node does not exist, create it
		if node.children[context[j]] == None:
			node.children[context[j]] = Node(m, context[i:j+1])
		# Update count
		node.count[nextSym] += 1
		node = node.children[context[j]]

	node.count[nextSym] += 1

def decreaseCount(root, context, nextSym):
	"""
	Decrease counts of nodes.
	Inputs:
		root:    root node of the tree
		context: context of next symbol
		nextSym: next symbol
	"""
	m       = root.m
	node    = root
	context = context[::-1]  # invert context

	# Navigate tree
	for j in range(len(context)):
		# Navigate through tree according to context
		if node.children[context[j]] != None:
			# Update count
			node.count[nextSym] -= 1
			node = node.children[context[j]]
		else:
			break

	node.count[nextSym] -= 1

def completeTree(node):
	"""
	Complete tree so that all non-leaf nodes have all children.
	Inputs:
		node: current tree node (start with root)
	"""
	m = node.m

	# If node is not leaf
	if not node.isLeaf():
		# If there is some child missing, add it
		for c in range(m):
			if node.children[c] == None:
				newContext = node.context.copy()
				newContext.append(c)
				node.children[c] = Node(m, newContext)
			completeTree(node.children[c])

def updateTree(root, data, D, type, beta=0.5, alpha=2, pruneFlag=True):
	"""
	Update tree by increasing counts relative to a given sequence.
	Inputs:
		root:      root node of tree
		data:      sequence of symbols
		D:         tree maximum depth
		type:      'CTW' or 'CTM'
		beta:      weighting coefficient
		pruneFlag: flag indicating whether to prune tree or not
	"""
	# For each symbol and corresponding context
	for tau in range(1,len(data)-D+1):
		start   = tau - 1
		context = data[start:start+D]
		nextSym = data[start+D]
		increaseCount(root, context, nextSym)

	completeTree(root)

	# Compute probabilities
	if type == 'CTW':
		root.getCTWProbs(D, beta=beta, alpha=alpha)
	elif type == 'CTM':
		root.getCTMProbs(D, beta=beta, alpha=alpha, pruneFlag=pruneFlag)
	else:
		print("ERROR: Type not recognised. Try 'CTW or 'CTM'.")
		return -1

	return root

def downgradeTree(root, data, D, type, beta=0.5, alpha=2, pruneFlag=False):
	"""
	Downgrade tree by decreasing counts relative to a given sequence.
	Inputs:
		root:      root node of tree
		data:      sequence of symbols
		D:         tree maximum depth
		type:      'CTW' or 'CTM'
		beta:      weighting coefficient
		pruneFlag: flag indicating whether to prune tree or not
	"""
	# For each symbol and corresponding context
	for tau in range(1,len(data)-D+1):
		start   = tau - 1
		context = data[start:start+D]
		nextSym = data[start+D]
		decreaseCount(root, context, nextSym)

	completeTree(root)

	# Compute probabilities
	if type == 'CTW':
		root.getCTWProbs(D, beta=beta, alpha=alpha)
	elif type == 'CTM':
		root.getCTMProbs(D, beta=beta, alpha=alpha, pruneFlag=pruneFlag)
	else:
		print("ERROR: Type not recognised. Try 'CTW or 'CTM'.")
		return -1

	return root

def pruneTree(root, D, beta=0.5):
	"""
	Prune tree according to current counts.
	Inputs:
		root:      root node of tree
		D:         tree maximum depth
		beta:      weighting coefficient
	"""
	root.getCTMProbs(D, beta, pruneFlag=True)
	return root

def getlog2ProbSeqPw(root):
	"""
	Return weighted probability of root node.
	"""
	return root.log2Pw

def getlog2ProbSeqPm(root):
	"""
	Return maximising probability of root node.
	"""
	return root.log2Pm
