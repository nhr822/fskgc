import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
from torch.autograd import Variable
from torch.nn.init import xavier_normal_


class ScaledDotProductAttention(nn.Module):
    def __init__(self, attn_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):     # q, k, v：[batch, time, dim] attn_mask: [batch, time]
        attn = torch.bmm(q, k.transpose(1, 2))
        if scale:
            attn = attn * scale
        if attn_mask:
            attn = attn.masked_fill_(attn_mask, -np.inf)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim, num_heads=8, dropout=0.0):
        super(MultiHeadAttention, self).__init__()

        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads

        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)

        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, query, key, value, attn_mask=None):       # 5, 2, 400

        residual = query
        batch_size = key.size(0)

        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        key = key.view(batch_size * self.num_heads, -1, self.dim_per_head)
        value = value.view(batch_size * self.num_heads, -1, self.dim_per_head)
        query = query.view(batch_size * self.num_heads, -1, self.dim_per_head)

        if attn_mask:
            attn_mask = attn_mask.repeat(self.num_heads, 1, 1)

        scale = (key.size(-1) // self.num_heads) ** -0.5
        context, attn = self.dot_product_attention(query, key, value, scale, attn_mask)
        context = context.view(batch_size, -1, self.dim_per_head * self.num_heads)
        output = self.linear_final(context)
        output = self.dropout(output)
        output = self.layer_norm(residual + output)
        return output, attn


class GELU(nn.Module):
    """
    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class PositionalWiseFeedForward(nn.Module):

    def __init__(self, model_dim=512, ffn_dim=2048, dropout=0.0):
        super(PositionalWiseFeedForward, self).__init__()
        self.w1 = nn.Conv1d(model_dim, ffn_dim, 1)
        self.w2 = nn.Conv1d(ffn_dim, model_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)
        self.gelu = GELU()

    def forward(self, x):           # [b, t, d*h]
        output = x.transpose(1, 2)  # [b, d*h, t]
        output = self.w1(output)
        # print(output.shape)
        output = self.w2(self.gelu(output))
        output = self.dropout(output.transpose(1, 2))
        # add residual and norm layer
        output = self.layer_norm(x + output)
        return output


class EncoderLayer(nn.Module):
    def __init__(self, model_dim=512, num_heads=8, ffn_dim=2048, dropout=0.0):
        # ffn_dim
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)
        self.model_dim = model_dim

    def forward(self, inputs, attn_mask=None):
        context, attention = self.attention(inputs, inputs, inputs, attn_mask)
        output = self.feed_forward(context)

        # output_e = output.view(-1, self.model_dim)
        # output_e = output_e.detach().numpy()  #
        # torch.set_printoptions(threshold=1000000000000)
        # file2 = open("output_5.txt", 'w')
        # file2.write(str(output_e))
        # file2.close()

        return output, attention


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_seq_len, device):
        super(PositionalEncoding, self).__init__()

        position_encoding = np.array([
            [pos / np.power(10000, 2.0 * (j // 2) / d_model) for j in range(d_model)]
            for pos in range(max_seq_len)])

        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])

        pad_row = torch.zeros([1, d_model], dtype=torch.float)
        position_encoding = torch.tensor(position_encoding, dtype=torch.float)

        position_encoding = torch.cat((pad_row, position_encoding), dim=0)

        self.position_encoding = nn.Embedding(max_seq_len + 1, d_model)
        self.position_encoding.weight = nn.Parameter(position_encoding, requires_grad=False)
        self.device = device

    def forward(self, batch_len, seq_len):
        input_pos = torch.tensor([list(range(1, seq_len + 1)) for _ in range(batch_len)]).to(self.device)
        return self.position_encoding(input_pos)            # [batch, time, dim]


class Att_GCN(nn.Module):
    def __init__(self, embed_dim, num_ents, num_rels, max_neighbor, device, use_pretrain=True, ent_embed=None,
                 dropout_input=0.3, finetune=False, dropout_neighbors=0.3, model_dim=100, ffn_dim=800, num_heads=4,
                 dropout=0.1, num_layers=6, max_seq_len=20, with_pos=True):

        super(Att_GCN, self).__init__()
        self.embed_dim = embed_dim
        self.ent_emb = nn.Embedding(num_ents, embed_dim)  # , padding_idx=self.num_ent+1
        self.rel_emb = nn.Embedding(num_rels, embed_dim)  # , padding_idx=self.num_ent+1

        self.max_neighbor = max_neighbor
        self.device = device

        self.fc_out = torch.nn.Linear(self.embed_dim, self.embed_dim)
        self.fc_self = torch.nn.Linear(self.embed_dim, self.embed_dim)

        self.fc = torch.nn.Linear(self.embed_dim*4, self.embed_dim)
        self.gcnfc = torch.nn.Linear(self.embed_dim, self.embed_dim)
        self.gcndrop = nn.Dropout(dropout_neighbors)
        self.gcnbn2 = torch.nn.BatchNorm1d(max_neighbor+1)     # 批规范化

        self.with_pos = with_pos
        self.num_heads = num_heads

        self.encoder_layers = nn.ModuleList([EncoderLayer(model_dim * num_heads, num_heads, ffn_dim, dropout)
                                             for _ in range(num_layers)])
        self.pos_embedding = PositionalEncoding(model_dim, max_seq_len, device)

    def init(self):
        xavier_normal_(self.ent_emb.weight.data)
        xavier_normal_(self.rel_emb.weight.data)

    def s_GCN(self, h_em, t_em, r_em, neb_e_em, neb_nebr_em, adj):
        adj = adj.float()                   # b/f, num_neb+1, num_neb+1
        # hrt_em = torch.cat([ht_em, r_em], 1)               # (b/f, 3, 100)

        ht_em = torch.cat([h_em, t_em], 1)      # (b/f, 2, 100)
        htr_em = torch.cat([ht_em, r_em], 1)      # (b/f, 3, 100)
        sum_hrt_em = htr_em.mean(1).unsqueeze(1)      # (b/f, 1, 100)

        neb_nebr_em_shuxing = neb_nebr_em.mean(2)           # (b/f, num_neb, 100)
        subg_em = (neb_e_em+neb_nebr_em_shuxing)/2.0       # (b/f, num_neb, 100)

        sum_hrt_em_ws = self.fc_self(sum_hrt_em)        # (b/f, 1, 100)
        subg_em_wo = self.fc_out(subg_em)        # 邻居实体  (b/f, num_neb, 100)
        subg_rd_em = torch.cat([sum_hrt_em_ws, subg_em_wo], 1)     # (b/f, num_neb+1, 100)

        pos = self.pos_embedding(batch_len=subg_rd_em.size(0), seq_len=self.max_neighbor+1)
        subg_rd_em_pos = subg_rd_em + pos

        subg_att_em_pos = subg_rd_em_pos.repeat(1, 1, self.num_heads)
        attentions = []
        for encoder in self.encoder_layers:
            subg_att_em_pos, attention = encoder(subg_att_em_pos)       # (b/f, num_neb+1, 400)
            attentions.append(attention)

        subg_att_em_pos = self.fc(subg_att_em_pos)           # b/f， num_neb+1， 100
        subg_att_em_pos = self.gcnfc(subg_att_em_pos)              # b/f， num_neb+1， 100
        gcn_em1 = torch.bmm(adj, subg_att_em_pos)        # b/f， num_neb+1， 100
        gcn_em1 = F.relu(gcn_em1)
        gcn_em1 = self.gcndrop(gcn_em1)
        gcn_em1 = self.gcnbn2(gcn_em1)

        gcn_em2 = torch.bmm(adj, gcn_em1)        # b/f， num_neb+1， 100
        gcn_em2 = F.relu(gcn_em2)
        gcn_em2 = self.gcndrop(gcn_em2)
        gcn_em2 = self.gcnbn2(gcn_em2)

        # tri_em = torch.mean(gcn_em2, 1)
        tri_em = gcn_em2[:, 0, :]   # b/f，100

        return tri_em

    def forward(self, hrt, neb, nebr, adj):     # b/f,3   b/f,num_neb    b/f,num_neb,b/f      5,num_neb+1,num_neb+1
        # ht = hrt[:, [0, 2]]     # 5,2

        h = hrt[:, 0]     # b/f,1
        t = hrt[:, 2]     # b/f,1
        r = hrt[:, 1]     # b/f,1

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


class Mapping(nn.Module):
    def __init__(self, device, model_dim=100, ffn_dim=800, num_heads=4, dropout=0.1, num_layers=6, max_seq_len=3,
                 with_pos=True):
        super(Mapping, self).__init__()
        self.fc_d1 = torch.nn.Linear(model_dim, model_dim)
        self.fc_d2 = torch.nn.Linear(model_dim, model_dim)

    def forward(self, h_em, t_em, r_embed, triple):
        entity_embed = h_em + t_em     # b/f， 1， 100
        entity_embed_d1 = self.fc_d1(entity_embed)      # b/f， 1， 100
        r_embed_d1 = self.fc_d1(r_embed)
        r_embed_d2 = self.fc_d2(r_embed)
        triple_embed_d2 = self.fc_d2(triple)

        rel_em_d1 = entity_embed_d1 + r_embed_d1        # b/f， 1， 100
        rel_em_d2 = r_embed_d2 + triple_embed_d2

        rel_em = torch.cat((rel_em_d1, rel_em_d2), dim=2)         # (b/f, 1, 200)
        rel_em = rel_em.squeeze(1)
        return rel_em


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()

    def forward(self, support, query):
        query_ = query.unsqueeze(1).expand(query.size(0), support.size(0), query.size(1)).contiguous()  # [b, few, dim]
        support_ = support.unsqueeze(0).expand_as(query_).contiguous()  # [b, few, dim]

        scalar = support.size(1) ** -0.5  # dim ** -0.5
        score = torch.sum(query_ * support_, dim=2) * scalar
        att = torch.softmax(score, dim=1)

        center = torch.mm(att, support)
        return center


class ACPrototype(nn.Module):
    def __init__(self, r_dim):
        super(ACPrototype, self).__init__()
        self.Attention = Attention(hidden_size=r_dim)

    def forward(self, support, query):
        center = self.Attention(support, query)
        return center
