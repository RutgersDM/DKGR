import numpy as np
from collections import defaultdict
import csv
import os


class RelationEntityAndClusterBatcher():
	def __init__(self, input_dir, batch_size, entity_vocab, relation_vocab, cluster_vocab, cluster_relation_vocab,
				 entity_id_to_cluster_mappping, mode="train"):
		self.input_dir = input_dir
		self.input_file = input_dir + '/{0}.txt'.format(mode)
		self.batch_size = batch_size

		print('Reading vocab...')
		self.entity_vocab = entity_vocab
		self.relation_vocab = relation_vocab
		self.cluster_vocab = cluster_vocab
		self.cluster_relation_vocab = cluster_relation_vocab

		self.entity_id_to_cluster_mappping = entity_id_to_cluster_mappping

		self.mode = mode
		self.create_triple_store(self.input_file)  # read raw data, including training and testing data
		print("batcher loaded")

	def get_next_batch(self):
		if self.mode == 'train':
			yield self.yield_next_batch_train()
		else:
			yield self.yield_next_batch_test()

	def create_triple_store(self, input_file):

		self.store_all_correct = defaultdict(set)
		self.store_cluster_all_correct = defaultdict(set)
		self.store = []

		if self.mode == 'train':
			with open(input_file) as raw_input_file:
				csv_file = csv.reader(raw_input_file, delimiter='\t')
				for line in csv_file:
					e1 = self.entity_vocab[line[0]]
					c1 = self.entity_id_to_cluster_mappping[str(e1)]
					r = self.relation_vocab[line[1]]
					e2 = self.entity_vocab[line[2]]
					c2 = self.entity_id_to_cluster_mappping[str(e2)]
					c_r = self.cluster_relation_vocab[str(c1) + '_' + str(c2)]

					self.store.append([e1, r, e2])
					self.store_all_correct[(e1, r)].add(e2)
					self.store_cluster_all_correct[(c1, c_r)].add(c2)
			self.store = np.array(self.store)
		else:
			with open(input_file) as raw_input_file:
				csv_file = csv.reader(raw_input_file, delimiter='\t')
				for line in csv_file:

					e1 = line[0]
					r = line[1]
					e2 = line[2]

					if e1 in self.entity_vocab and e2 in self.entity_vocab:

						e1 = self.entity_vocab[e1]
						r = self.relation_vocab[r]
						e2 = self.entity_vocab[e2]

						self.store.append([e1, r, e2])
			self.store = np.array(self.store)
			fact_files = ['train.txt', 'test.txt', 'dev.txt', 'graph.txt']
			if os.path.isfile(self.input_dir + '/' + 'full_graph.txt'):
				fact_files = ['full_graph.txt']
				print("Contains full graph")

			for f in fact_files:
				with open(self.input_dir + '/' + f) as raw_input_file:
					csv_file = csv.reader(raw_input_file, delimiter='\t')
					for line in csv_file:

						e1 = line[0]
						r = line[1]
						e2 = line[2]

						if e1 in self.entity_vocab and e2 in self.entity_vocab:
							e1 = self.entity_vocab[e1]
							c1 = self.entity_id_to_cluster_mappping[str(e1)]
							r = self.relation_vocab[r]
							e2 = self.entity_vocab[e2]
							c2 = self.entity_id_to_cluster_mappping[str(e2)]

							self.store_all_correct[(e1, r)].add(e2)
							c_r = self.cluster_relation_vocab[str(c1) + '_' + str(c2)]
							self.store_cluster_all_correct[(c1, c_r)].add(c2)

	def yield_next_batch_train(self):
		while True:
			batch_idx = np.random.randint(0, self.store.shape[0], size=self.batch_size)
			batch = self.store[batch_idx, :]
			e1 = batch[:, 0]
			c1 = np.array([self.entity_id_to_cluster_mappping[str(e)] for e in e1])

			r = batch[:, 1]
			e2 = batch[:, 2]
			c2 = np.array([self.entity_id_to_cluster_mappping[str(e)] for e in e2])
			c_r = np.array([str(c1[i]) + '_' + str(c2[i]) for i in range(c1.shape[0])])

			all_e2s = []
			all_c2s = []
			for i in range(e1.shape[0]):
				all_e2s.append(self.store_all_correct[(e1[i], r[i])])
			for i in range(c1.shape[0]):
				all_c2s.append(self.store_cluster_all_correct[(c1[i], c_r[i])])

			assert e1.shape[0] == e2.shape[0] == r.shape[0] == len(all_e2s)
			assert c1.shape[0] == c2.shape[0] == c_r.shape[0] == len(all_c2s)

			yield (e1, r, e2, all_e2s), (c1, c2, all_c2s)

	def yield_next_batch_test(self):
		remaining_triples = self.store.shape[0]
		current_idx = 0
		while True:
			if remaining_triples == 0:
				return

			if remaining_triples - self.batch_size > 0:
				batch_idx = np.arange(current_idx, current_idx + self.batch_size)
				current_idx += self.batch_size
				remaining_triples -= self.batch_size
			else:
				batch_idx = np.arange(current_idx, self.store.shape[0])
				remaining_triples = 0

			batch = self.store[batch_idx, :]
			e1 = batch[:, 0]
			c1 = np.array([self.entity_id_to_cluster_mappping[str(e)] for e in e1])

			r = batch[:, 1]
			e2 = batch[:, 2]
			c2 = np.array([self.entity_id_to_cluster_mappping[str(e)] for e in e2])
			c_r = np.array([str(c1[i]) + '_' + str(c2[i]) for i in range(c1.shape[0])])

			all_e2s = []
			all_c2s = []
			for i in range(e1.shape[0]):
				all_e2s.append(self.store_all_correct[(e1[i], r[i])])
			for i in range(c1.shape[0]):
				all_c2s.append(self.store_cluster_all_correct[(c1[i], c_r[i])])

			assert e1.shape[0] == e2.shape[0] == r.shape[0] == len(all_e2s)
			assert c1.shape[0] == c2.shape[0] == c_r.shape[0] == len(all_c2s)

			yield (e1, r, e2, all_e2s), (c1, c2, all_c2s)
