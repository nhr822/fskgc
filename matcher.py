import logging

import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_

from modules import *


class EntityEncoder(nn.Module):
    def __init__(self, embed_dim, num_ents, num_rels, max_neighbor, device, use_pretrain=True, ent_embed=None,
                 dropout_input=0.3, finetune=False, dropout_neighbors=0.3):
        super(EntityEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.ent_emb = nn.Embedding(num_ents, embed_dim)  # , padding_idx=self.num_ent+1
        self.rel_emb = nn.Embedding(num_rels, embed_dim)  # , padding_idx=self.num_ent+1

        self.max_neighbor = max_neighbor
        self.device = device

        self.att_fc = torch.nn.Linear(self.embed_dim, self.embed_dim)

        self.gcnfc = torch.nn.Linear(self.embed_dim, self.embed_dim)
        self.gcndrop = nn.Dropout(dropout_neighbors)
        self.gcnbn2 = torch.nn.BatchNorm1d(max_neighbor+1)     # 批规范化

    def init(self):
        xavier_normal_(self.ent_emb.weight.data)
        xavier_normal_(self.rel_emb.weight.data)

    def s_GCN(self, h_em, t_em, r_em, neb_e_em, neb_nebr_em, adj):
        adj = adj.float()                   # b/f, num_neb+1, num_neb+1
        # hrt_em = torch.cat([ht_em, r_em], 1)               # (b/f, 3, 100)

        ht_em = torch.cat([h_em, t_em], 1)      # (b/f, 2, 100)
        htr_em = torch.cat([ht_em, r_em], 1)      # (b/f, 3, 100)
        sum_hrt_em = htr_em.mean(1).unsqueeze(1)      # (b/f, 1, 100)

        neb_nebr_em_shuxing = neb_nebr_em.mean(2)           # (b/f, 19, 100)
        subg_em = (neb_e_em+neb_nebr_em_shuxing)/2.0       # (b/f, 19, 100)

        sum_hrt_em_w = self.att_fc(sum_hrt_em)        # (b/f, 1, 100)
        subg_em_w = self.att_fc(subg_em)        # 邻居实体  (b/f, 19, 100)
        subg_em_w = subg_em_w.transpose(2,1)
        att = torch.bmm(sum_hrt_em_w, subg_em_w)       # (b/f, 1, 19)
        att = att.squeeze(1)            # (b/f, 19)
        att = F.leaky_relu(att)
        att = F.softmax(att)
        att = att.unsqueeze(2)      # (b/f, 19, 1)

        subg_em_em = att*subg_em     # (b/f, 19, 100)

        subg_att_em = torch.cat([sum_hrt_em, subg_em_em], 1)     # (b/f, 20, 100)

        subg_att_em = self.gcnfc(subg_att_em)              # b/f， 20， 100
        gcn_em1 = torch.bmm(adj, subg_att_em)        # b/f， 20， 100
        gcn_em1 = F.relu(gcn_em1)
        gcn_em1 = self.gcndrop(gcn_em1)
        gcn_em1 = self.gcnbn2(gcn_em1)

        gcn_em2 = torch.bmm(adj, gcn_em1)        # b/f， 20， 100
        gcn_em2 = F.relu(gcn_em2)
        gcn_em2 = self.gcndrop(gcn_em2)
        gcn_em2 = self.gcnbn2(gcn_em2)

        # tri_em = torch.mean(gcn_em2, 1)
        tri_em = gcn_em2[:, 0, :]   # b/f，100

        return tri_em

    def forward(self, hrt, neb, nebr, adj): # 5,3   5,19    5,19,5      5,20,20
        # ht = hrt[:, [0, 2]]     # 5,2

        h = hrt[:, 0]     # 5,1
        t = hrt[:, 2]     # 5,1
        r = hrt[:, 1]     # 5,1

        h_em = self.ent_emb(h)            # (b/f, 100)
        t_em = self.ent_emb(t)            # (b/f, 100)
        r_em = self.rel_emb(r)              # (b/f, 100)
        h_em = h_em.unsqueeze(1)            # (b/f, 1, 100)
        t_em = t_em.unsqueeze(1)            # (b/f, 1, 100)
        r_em = r_em.unsqueeze(1)            # (b/f, 1, 100)

        neb_e_em = self.ent_emb(neb)        # (b/f, num_neb， 100)
        neb_nebr_em = self.rel_emb(nebr)        # (b/f, num_neb, num_nebr， 100)

        tri_em = self.s_GCN(h_em, t_em, r_em, neb_e_em, neb_nebr_em, adj)
        tri_em = tri_em.unsqueeze(1)            # (b/f, 1, 100)

        return h_em, t_em, r_em, tri_em


class RelationRepresentation(nn.Module):
    def __init__(self, emb_dim, device, num_transformer_layers, num_transformer_heads, dropout_rate=0.1):
        super(RelationRepresentation, self).__init__()
        self.RelationEncoder = TransformerEncoder(device, model_dim=emb_dim,
                                                  ffn_dim=emb_dim * num_transformer_heads * 2,
                                                  num_heads=num_transformer_heads, dropout=dropout_rate,
                                                  num_layers=num_transformer_layers, max_seq_len=3, with_pos=True)

    def forward(self, h_em, t_em, r_em, triple_ht):
        relation = self.RelationEncoder(h_em, t_em, r_em, triple_ht)
        return relation


class sub_GCN(nn.Module):
    def __init__(self, embed_dim, num_ents, num_rels, max_neighbor, device, use_pretrain=True, ent_embed=None,
                 dropout_layers=0.1, dropout_input=0.3, dropout_neighbors=0.0, finetune=False, num_transformer_layers=6,
                 num_transformer_heads=4):
        super(sub_GCN, self).__init__()
        self.EntityEncoder = EntityEncoder(embed_dim, num_ents, num_rels, max_neighbor, device,
                                           use_pretrain=use_pretrain, ent_embed=ent_embed, dropout_input=dropout_input,
                                           dropout_neighbors=dropout_neighbors, finetune=finetune)
        self.RelationRepresentation = RelationRepresentation(embed_dim, device,
                                                             num_transformer_layers=num_transformer_layers,
                                                             num_transformer_heads=num_transformer_heads,
                                                             dropout_rate=dropout_layers)
        self.Prototype = SoftSelectPrototype(embed_dim * num_transformer_heads)

    def forward(self, hrt_list, neb_list, nebr_list, adj_list, isEval=False):
        if not isEval:
            s_h_em, s_t_em, s_r_em, support_tri = self.EntityEncoder(hrt_list[0], neb_list[0], nebr_list[0], adj_list[0])
            q_h_em, q_t_em, q_r_em, query_tri = self.EntityEncoder(hrt_list[1], neb_list[1], nebr_list[1], adj_list[1])
            f_h_em, f_t_em, f_r_em, false_tri = self.EntityEncoder(hrt_list[2], neb_list[2], nebr_list[2], adj_list[2])

            support_r = self.RelationRepresentation(s_h_em, s_t_em, s_r_em, support_tri)        # 5, 400
            query_r = self.RelationRepresentation(q_h_em, q_t_em,  q_r_em, query_tri)
            false_r = self.RelationRepresentation(f_h_em, f_t_em,  f_r_em, false_tri)

            center_q = self.Prototype(support_r, query_r)
            center_f = self.Prototype(support_r, false_r)

            positive_score = torch.sum(query_r * center_q, dim=1)
            negative_score = torch.sum(false_r * center_f, dim=1)
        else:

            s_h_em, s_t_em, s_r_em, support_tri = self.EntityEncoder(hrt_list[0], neb_list[0],nebr_list[0], adj_list[0])
            q_h_em, q_t_em, q_r_em, query_tri = self.EntityEncoder(hrt_list[1], neb_list[1],nebr_list[1], adj_list[1])

            support_r = self.RelationRepresentation(s_h_em, s_t_em, s_r_em,support_tri)        # 5, 400
            query_r = self.RelationRepresentation(q_h_em, q_t_em, q_r_em, query_tri)

            center_q = self.Prototype(support_r, query_r)

            positive_score = torch.sum(query_r * center_q, dim=1)
            negative_score = None
        return positive_score, negative_score
