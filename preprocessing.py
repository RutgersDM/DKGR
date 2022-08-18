def K_means_clustering(K, dataPath='../FB15k-237/', entity2idPath='entity2id.txt', relation2idPath='relation2id.txt',
					   pretrainedEmbPath='entity2vec.bern', graphPath='graph.txt'):
	# read entity IDs
	entity2idPath = os.path.join(dataPath, entity2idPath)

	with open(entity2idPath, "r") as f:
		entity2id = {}
		for line in f:
			entity, eid = line.split()
			entity2id[entity] = int(eid)

	# read relation IDs
	relation2idPath = os.path.join(dataPath, relation2idPath)

	with open(relation2idPath, "r") as f:
		relation2id = {}
		for line in f:
			relation, rid = line.split()
			relation2id[relation] = int(rid)

	# read entity embeddings
	pretrained_emb_file = os.path.join(dataPath, pretrainedEmbPath)

	entity2emb = []
	with open(pretrained_emb_file, "r") as f:
		for line in f:
			entity2emb.append([float(value) for value in line.split()])

	# entity2emb = np.load(pretrained_emb_file)
	# entity2emb = list(entity2emb)

	# K Means CLustering
	kmeans_entity = KMeans(n_clusters=K, random_state=0).fit(entity2emb)

	# assign cluster label to entities
	entity2cluster = {}

	for idx, label in enumerate(kmeans_entity.labels_):
		entity2cluster[idx] = int(label)

	# print(entity2cluster)

	ent2clusterFile = os.path.join(dataPath, 'entity2clusterid.txt')
	with open(ent2clusterFile, 'w') as f:
		for ent in entity2cluster.keys():
			f.write(str(ent) + '\t' + str(entity2cluster[ent]) + '\n')