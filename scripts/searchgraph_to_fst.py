#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Takes a single Moses search graph on STDIN and writes out a corresponding FST and dictionary.
"""

import sys
import codecs
import argparse

reload(sys)
sys.setdefaultencoding('utf-8')
sys.stdin = codecs.getreader('utf-8')(sys.stdin)
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)
sys.stdout.encoding = 'utf-8'

parser = argparse.ArgumentParser()
parser.add_argument('--prefix', type=str, default='graph',
                    help="prefix to use in naming output graphs")
parser.add_argument('--input', type=str, default=None,
                    help="The file with corresponding input sentences")
args = parser.parse_args()

# to run: python searchgraph_to_fst.py [graph file] 
# to view the graph: use graphvis gui 

FST = False
if args.input is not None:
        FST = True
	input_text = codecs.open(args.input,'rb','utf-8').readlines()

graph_file = sys.stdin

#get all the edges in a string
edgelist=""
end_nodes=[]
nodes=[]


#create an indexer for all the labels
indexer = {}
indexer["<eps>"]=0
max_index=0



def get_index(word):
	global max_index
	global indexer
	if word in indexer:
		return indexer[word]
	else: 
		max_index+=1
		indexer[word]=max_index
		return indexer[word]

def print_string_indices():
	out_indicies=""
	for key in indexer:
		out_indicies+=key +" "+ str(indexer[key]) + "\n"

	keys_file.write(out_indicies)



line = graph_file.readline()
line_num = int(line[0: line.find(" ")])

while not line =='':
        e_start = line.find("recombined=")
	if e_start <0: # recombined not found, just use hype node
		e_start = line.find("hyp=")+4 
		e_end =  line.find(" ",e_start)
		e = line[e_start:e_end ]
	else:
		e_start +=11
		e_end =  line.find(" ",e_start)
		e = line[e_start:e_end ]


	s_start = line.find("back=") 
	if s_start <0: # back not found, this is just the start node	
		line = graph_file.readline()
		continue
	s_start += 5 
	s_end =  line.find(" ",s_start)
	s = line[s_start:s_end ]

	weight_start = line.find("transition=") +11
	weight_end = line.find(" ",weight_start)
	weight=line[weight_start:weight_end]


	#check if this is an end node
	end_start = line.find("forward=") +8
	end_end = line.find(" ",end_start)
	end=line[end_start:end_end]
	if end=="-1":
		end_nodes+= [int(e)]

	nodes+= [int(e)] + [int(s)]



	label_start = line.find("out=")+4 
	label_end = line.find("\n")
	label = line[label_start:label_end ]

	#add label to hash table
	label=label.replace(" ", "|")
	get_index(label)
	
	if FST:
		input_text_line=input_text[int(line_num)].split()
		coverage_1_start=line.find("covered=") +8
		coverage_1_end=  line.find("-",coverage_1_start)
		coverage_1 = line[coverage_1_start: coverage_1_end]

		coverage_2_start=coverage_1_end +1
		coverage_2_end=  line.find(" ",coverage_2_start)
		coverage_2 = line[coverage_2_start: coverage_2_end]

		coverage =  "|".join(input_text_line[int(coverage_1): int(coverage_2)+1])
		get_index(coverage)

		edgelist+= str(s) +" " + str(e) +" "+ coverage +" "+ label + " "+ str(-1*float(weight))+"\n"
	else:
		edgelist+= str(s) +" " + str(e) +" "+ label +" "+ label + " "+ str(-1*float(weight))+"\n"

	#advance loop
	line = graph_file.readline()
	

###Write out the files
fout_name = "{}.{}.fst.txt".format(args.prefix, line_num)
fout_keys_name = "{}.{}".format(args.prefix, line_num) +  ".keys"
fst_file=codecs.open(fout_name,'wb','utf-8')
keys_file=codecs.open(fout_keys_name,'wb','utf-8')


nodes=set(nodes) 
end_nodes=set(end_nodes)

end = max(nodes)

 # add endstate for all nodes that dont have one
end +=1
for node in end_nodes:
	get_index("</eos>")
 	edgelist+= str(node) + " " + str(end) + " </eos> </eos> 0\n"

edgelist+= str(end)


#write edgelist
fst_file.write(edgelist)

#make the keys file for the symbol table 
print_string_indices()
