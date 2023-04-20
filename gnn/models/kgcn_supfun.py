# -*- coding: utf-8 -*-
"""
Created on Thu May  5 23:39:47 2022
@author: QI YU
@email: yq123456leo@outlook.com
"""

import numpy as np

class test:
    def get_receptive_field(self, entity):
        neigh_ent_list = [entity]
        neigh_rel_list = []
    
        adj_entity_matrix = self.config.adj_entity
        adj_relation_matrix = self.config.adj_relation
        n_neighbor = adj_entity_matrix.shape[1]

        for i in range(self.config.n_depth):
            new_neigh_ent = adj_entity_matrix[neigh_ent_list[-1].astype('int64')]
            new_neigh_rel = adj_relation_matrix[neigh_ent_list[-1].astype('int64')]
            
            neigh_ent_list.append(
                np.reshape(new_neigh_ent, (-1, n_neighbor ** (i + 1))))
            neigh_rel_list.append(
                np.reshape(new_neigh_rel, (-1, n_neighbor ** (i + 1))))

        return neigh_ent_list + neigh_rel_list

    def get_neighbor_info(self, embeddings):
        drug, rel, ent = embeddings
        
        # [batch_size, neighbor_size ** hop, 1] drug-entity score
        drug_rel_score = np.sum(drug * rel, axis = -1, keepdims = True)

        # [batch_size, neighbor_size ** hop, embed_dim]
        weighted_ent = drug_rel_score * ent

        # [batch_size, neighbor_size ** (hop-1), neighbor_size, embed_dim]
        weighted_ent = np.reshape(weighted_ent,
                                  (weighted_ent.shape[0], -1,
                                   self.config.neighbor_sample_size, self.config.embed_dim))

        neighbor_embed = np.sum(weighted_ent, axis = 2)
        return neighbor_embed