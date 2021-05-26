from collections import defaultdict
import logging
import numpy as np
import csv

logger = logging.getLogger(__name__)

class RelationEntityGrapher:
    def __init__(self, triple_store, relation_vocab, entity_vocab, max_num_actions, entity_id_to_cluster_mappping):

        self.ePAD = entity_vocab['PAD']
        self.rPAD = relation_vocab['PAD']
        self.triple_store = triple_store
        self.relation_vocab = relation_vocab
        self.entity_vocab = entity_vocab
        self.store = defaultdict(list)
        self.array_store = np.ones((len(entity_vocab), max_num_actions, 2)).astype(int)
        self.array_store[:, :, 0] *= self.ePAD
        self.array_store[:, :, 1] *= self.rPAD
        self.entity_id_to_cluster_mappping = entity_id_to_cluster_mappping

        self.masked_array_store = None

        self.rev_relation_vocab = dict([(v, k) for k, v in relation_vocab.items()])
        self.rev_entity_vocab = dict([(v, k) for k, v in entity_vocab.items()])

        self.create_graph()
        print("KG constructed")



    def create_graph(self):
        with open(self.triple_store) as triple_file_raw:
            triple_file = csv.reader(triple_file_raw, delimiter='\t')
            for line in triple_file:

                e1 = self.entity_vocab[line[0]]
                r = self.relation_vocab[line[1]]
                e2 = self.entity_vocab[line[2]]

                self.store[e1].append((r, e2))

        for e1 in self.store:
            num_actions = 1
            self.array_store[e1, 0, 1] = self.relation_vocab['NO_OP']
            self.array_store[e1, 0, 0] = e1
            for r, e2 in self.store[e1]:
                if num_actions == self.array_store.shape[1]: # default threshold is 200
                    break
                self.array_store[e1,num_actions,0] = e2
                self.array_store[e1,num_actions,1] = r
                num_actions += 1


        # save memory
        del self.store
        self.store = None

    def return_next_actions(self, current_entities, start_entities, query_relations, answers, all_correct_answers, last_step, rollouts, cluster_path, e_agent_cls_path, p_len):
        ret = self.array_store[current_entities, :, :].copy()
        whether_e_agent_follows_c_agent = []
        # cluster_path: dict[triple_id] = [c1, c2, ...]
        # cnt = 0
        for i in range(current_entities.shape[0]):  # original batch_size * 3 * num_rollout, current_entities = [...], 1-D

            if current_entities[i] == start_entities[i]: # mask out all direct answers
                relations = ret[i, :, 1]
                entities = ret[i, :, 0]
                mask = np.logical_and(relations == query_relations[i], entities == answers[i])
                ret[i, :, 0][mask] = self.ePAD
                ret[i, :, 1][mask] = self.rPAD

            if current_entities[i] == 0:
                e_agent_cls_path[p_len].append(0)
            else:
                e_agent_cls_path[p_len].append(self.entity_id_to_cluster_mappping[str(current_entities[i])])

            if last_step:
                entities = ret[i, :, 0]
                relations = ret[i, :, 1]

                correct_e2 = answers[i]
                for j in range(entities.shape[0]):
                    if entities[j] in all_correct_answers[int(i/rollouts)] and entities[j] != correct_e2:
                        entities[j] = self.ePAD
                        relations[j] = self.rPAD


            # print(current_entities[i], i, current_entities.shape[0])

            # if current_entities[i] not in self.entity_id_to_cluster_mappping:
            #     whether_e_agent_follows_c_agent.append(1.0)
            #     ret[i, :, 0] = self.ePAD
            #     ret[i, :, 1] = self.rPAD
            #     # print(ret[i, :, :])
            #     continue

            # if self.entity_id_to_cluster_mappping[str(current_entities[i])] == cluster_path[i][0]: # mask out the unfollowed entity-level answers
            #     whether_e_agent_follows_c_agent.append(0.0)
            # elif self.entity_id_to_cluster_mappping[str(current_entities[i])] == cluster_path[i][1]:
            #     whether_e_agent_follows_c_agent.append(0.0)
            #     del cluster_path[i][:1]
            # else:
            #     whether_e_agent_follows_c_agent.append(1.0)
            #     ret[i, :, 0] = self.ePAD
            #     ret[i, :, 1] = self.rPAD

        # print('The number of break-out agents: ', cnt)

        return ret, cluster_path, whether_e_agent_follows_c_agent, e_agent_cls_path

    def init_actions(self, current_entities, start_entities, query_relations, answers, all_correct_answers, last_step, rollouts):
        ret = self.array_store[current_entities, :, :].copy()
        # cluster_path: dict[triple_id] = [c1, c2, ...]
        # cnt = 0
        for i in range(current_entities.shape[0]):  # original batch_size * 3 * num_rollout, current_entities = [...], 1-D
            if current_entities[i] == start_entities[i]:  # mask out all direct answers
                relations = ret[i, :, 1]
                entities = ret[i, :, 0]
                mask = np.logical_and(relations == query_relations[i], entities == answers[i])
                ret[i, :, 0][mask] = self.ePAD
                ret[i, :, 1][mask] = self.rPAD
        return ret


class RelationClusterGrapher:
    def __init__(self, cluster_relation_vocab, cluster_vocab):

        self.ePAD = cluster_vocab['PAD']
        # self.rPAD = cluster_relation_vocab['PAD']

        self.cluster_relation_vocab = cluster_relation_vocab
        self.cluster_vocab = cluster_vocab

        self.array_store = np.ones((len(cluster_vocab), len(cluster_vocab), 2)).astype(int)
        self.array_store[:, :, 0] *= self.ePAD

        self.store = defaultdict(list)

        self.masked_array_store = None

        self.rev_cluster_relation_vocab = dict([(v, k) for k, v in cluster_relation_vocab.items()])
        self.rev_cluster_vocab = dict([(v, k) for k, v in cluster_vocab.items()])
        self.create_graph()

    def create_graph(self):

        for c_rel, rel_index in self.cluster_relation_vocab.items():
            if c_rel.split('_')[0].isdigit():
                c1, c2 = c_rel.split('_')
                self.store[int(c1)].append((rel_index, int(c2)))

        for c1 in self.store:
            num_actions = 1
            self.array_store[c1, 0, 1] = self.cluster_relation_vocab[str(c1)+'_'+str(c1)]
            self.array_store[c1, 0, 0] = c1
            for r, c2 in self.store[c1]:
                if num_actions == self.array_store.shape[1]: # default threshold is 75
                    break
                self.array_store[c1,num_actions,0] = c2
                self.array_store[c1,num_actions,1] = r
                num_actions += 1

        # save memory
        del self.store
        self.store = None

    def return_next_actions_cluster(self, current_clusters, start_clusters, answers, all_correct_answers, last_step, rollouts):
        ret = self.array_store[current_clusters, :, :].copy()
        return ret