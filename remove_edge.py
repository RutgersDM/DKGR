import json
import numpy as np
import torch
import csv

kg = 'datasets/data_preprocessed/teamplayssport/original_graph.txt'

removed_edges1 = 'datasets/data_preprocessed/teamplayssport/sparsity_removed_edge1'
removed_edges2 = 'datasets/data_preprocessed/teamplayssport/sparsity_removed_edge2'
removed_edges3 = 'datasets/data_preprocessed/teamplayssport/sparsity_removed_edge3'
removed_edges4 = 'datasets/data_preprocessed/teamplayssport/sparsity_removed_edge4'
removed_edges5 = 'datasets/data_preprocessed/teamplayssport/sparsity_removed_edge5'

removed_kg_file = 'datasets/data_preprocessed/teamplayssport/graph.txt'


f = open(removed_edges1)
remove_edges1 = f.readlines()
f = open(removed_edges2)
remove_edges2 = f.readlines()
f = open(removed_edges3)
remove_edges3 = f.readlines()
f = open(removed_edges4)
remove_edges4 = f.readlines()
f = open(removed_edges5)
remove_edges5 = f.readlines()

remove_edges = remove_edges1 \
			   # + remove_edges2 + remove_edges3 + remove_edges4 + remove_edges5

remove_edges = [edge.strip().split() for edge in remove_edges]

removed_kg = []
with open(kg) as triple_file_raw:
	triple_file = csv.reader(triple_file_raw, delimiter='\t')

	for line in triple_file:
		e1, r, e2 = line
		if [e1, r, e2] not in remove_edges:
			removed_kg.append(line)

with open(removed_kg_file, 'w') as f:
	for line in removed_kg:
		f.write('\t'.join(line) + '\n')


