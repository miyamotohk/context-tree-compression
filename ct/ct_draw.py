################################################
# Context-tree functions
################################################

import os

### Main function
def drawTree(root, type, filename='tree', show_probs=False):
	"""
	Draw a context-tree as a LaTeX file.
	Inputs:
		root:       root node of the tree
		type:       'CTW' or 'CTM'
		filename:   filename
		show_probs: flag indicating whether to show probability on tree nodes or not
	"""
	if type == 'CTW':
		drawCTWTree(root, filename, show_probs)
	elif type == 'CTM':
		drawCTMTree(root, filename, show_probs)
	else:
		print("ERROR: Type not recognised. Try 'CTW' or 'CTM'.")

### CTW
def drawCTWTree(root, filename='ctw_tree', show_probs=False):
	"""
	Create and compile TeX file to represent CTW tree.
	"""
	node = root
	file = open('{}.tex'.format(filename), 'w')
	# Write header
	file.writelines(
		["\\documentclass[tikz,border=10pt]{standalone}\n",
		"\\usepackage[linguistics]{forest}\n",
		"\\begin{document}\n",
		"\\begin{forest}\n",
		"for tree={grow=west}\n"]
		)
	
	file.write("[")
	drawCTWNodes(file,root, show_probs)
	file.write("]\n")

	file.writelines(
		["\\end{forest}\n",
		"\\end{document}"]
		)
	file.close()

	os.system("pdflatex -interaction=batchmode {}.tex".format(filename))

def drawCTWNodes(file, node, show_probs=False):
	"""
	Add each CTW node to TeX file.
	"""
	if node.label == '':
		file.write("{{$\\lambda$, {}\\\\ $P_e$={}\\\\ $P_w$={}}}\n".format(node.count,2**node.log2Pe,2**node.log2Pw))
	else:
		file.write("{{`{}\', {}\\\\ $P_e$={}\\\\ $P_w$={}}}\n".format(node.label,node.count,2**node.log2Pe,2**node.log2Pw))
	for child in node.children:
		if child != None:
			file.write("[")
			if child.isLeaf():
				None
				file.write("{{`{}\', {}\\\\ $P_e$={}\\\\ $P_w$={}}}\n".format(child.label,child.count,2**child.log2Pe,2**child.log2Pw))
			else:
				drawCTWNodes(file,child)
			file.write("]\n")

### CTM
def drawCTMTree(root, filename='ctm_tree', show_probs=False):
	"""
	Create and compile TeX file to represent CTM tree.
	"""
	node = root
	file = open('{}.tex'.format(filename), 'w')
	# Write header
	file.writelines(
		["\\documentclass[tikz,border=10pt]{standalone}\n",
		"\\usepackage[linguistics]{forest}\n",
		"\\begin{document}\n",
		"\\begin{forest}\n",
		"for tree={grow=west}\n"]
		)
	
	file.write("[")
	drawCTMNodes(file,root, show_probs)
	file.write("]\n")

	file.writelines(
		["\\end{forest}\n",
		"\\end{document}"]
		)
	file.close()

	os.system("pdflatex -interaction=batchmode {}.tex".format(filename))

def drawCTMNodes(file, node, show_probs=False):
	"""
	Add each CTW node to TeX file.
	"""
	if node.label == '':
		if show_probs==False:
			file.write("{{$\\lambda$\n }}\n")
		else:
			file.write("{{$\\lambda$, {}\\\\ $P_e$={}\\\\ $P_m$={}}}\n".format(node.count,2**node.log2Pe,2**node.log2Pm))
	else:
		if show_probs==False:
			file.write("{{`{}\'}}\n".format(node.label))
		else:
			file.write("{{`{}\', {}\\\\ $P_e$={}\\\\ $P_m$={}}}\n".format(node.label,node.count,2**node.log2Pe,2**node.log2Pm))
	for child in node.children:
		if child != None:
			file.write("[")
			if child.isLeaf():
				if show_probs==False:
					file.write("{{`{}\'}}\n".format(child.label))
				else:
					file.write("{{`{}\', {}\\\\ $P_e$={}\\\\ $P_m$={}}}\n".format(child.label,child.count,2**child.log2Pe,2**child.log2Pm))
			else:
				drawCTMNodes(file,child,show_probs)
			file.write("]\n")