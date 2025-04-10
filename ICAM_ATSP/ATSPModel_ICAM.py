

import torch
import torch.nn as nn
import torch.nn.functional as F



class ATSPModel(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params

        self.encoder = ATSP_Encoder(**model_params)
        self.decoder = ATSP_Decoder(**model_params)

        self.encoded_row = None
        self.encoded_col = None
        # shape: (batch, node, embedding)
        self.log_scale = None

        self.k = self.model_params['neighbors']
        embedding_dim = self.model_params['embedding_dim']
        self.embedding_row = nn.Linear(self.k, embedding_dim)
        self.embedding_col = nn.Linear(self.k, embedding_dim)

    def set_decoder_type(self,decoder_type):
        self.model_params['eval_type'] = decoder_type


    def pre_forward(self, reset_state):

        problems = reset_state.problems
        # problems.shape: (batch, node, node)
        self.log_scale = reset_state.log_scale

        problems_k_row = torch.topk(problems, k=self.k, dim=2, largest=False, sorted=True).values
        problems_k_col = torch.topk(problems.transpose(1, 2), k=self.k, dim=2, largest=False, sorted=True).values
        # shape: (batch, node, k)

        row_emb = self.embedding_row(problems_k_row)
        col_emb = self.embedding_col(problems_k_col)
        # shape: (batch, node, embedding)

        self.encoded_row, self.encoded_col = self.encoder(row_emb, col_emb, problems,self.log_scale)
        # encoded_nodes.shape: (batch, node, embedding)

        self.decoder.set_kv(self.encoded_col)


    def forward(self, state,cur_dist):

        batch_size = state.batch_size
        pomo_size = state.pomo_size

        if state.current_node is None:
            selected = torch.arange(pomo_size)[None, :].expand(batch_size, pomo_size)
            prob = torch.ones(size=(batch_size, pomo_size))

            encoded_first_row = _get_encoding(self.encoded_row, selected)
            # shape: (batch, pomo, embedding)
            self.decoder.set_q1(encoded_first_row)

        else:
            encoded_current_row = _get_encoding(self.encoded_row, state.current_node)
            # shape: (batch, pomo, embedding)
            all_job_probs = self.decoder(encoded_current_row,
                                         cur_dist,
                                         self.log_scale,
                                         ninf_mask=state.ninf_mask)
            # shape: (batch, pomo, job)
            assert not torch.isnan(all_job_probs).any(), "probs has nan, but it should not have any nans."

            if self.training or self.model_params['eval_type'] == 'sampling':
                while True:  # to fix pytorch.multinomial bug on selecting 0 probability elements
                    with torch.no_grad():
                        selected = all_job_probs.reshape(batch_size * pomo_size, -1).multinomial(1) \
                            .squeeze(dim=1).reshape(batch_size, pomo_size)
                        # shape: (batch, pomo)

                    prob = torch.gather(all_job_probs, dim=-1, index=selected.unsqueeze(-1)).squeeze(dim=-1)
                    # shape: (batch, pomo)

                    if (prob != 0).all():
                        break

            elif self.model_params['eval_type'] == 'greedy':
                selected = all_job_probs.argmax(dim=-1)
                # shape: (batch, pomo)
                prob = None
            else:
                raise NotImplementedError(f"eval_type: {self.model_params['eval_type']} is not implemented!")

        return selected, prob


def _get_encoding(encoded_nodes, node_index_to_pick):
    # encoded_nodes.shape: (batch, problem, embedding)
    # node_index_to_pick.shape: (batch, pomo)

    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)

    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
    # shape: (batch, pomo, embedding)

    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
    # shape: (batch, pomo, embedding)

    return picked_nodes


########################################
# ENCODER
########################################
class ATSP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        encoder_layer_num = model_params['encoder_layer_num']
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])

    def forward(self, row_emb, col_emb, cost_mat, log_scale):
        # col_emb.shape: (batch, col_cnt, embedding)
        # row_emb.shape: (batch, row_cnt, embedding)
        # cost_mat.shape: (batch, row_cnt, col_cnt)

        cost_mat = -1 * log_scale * cost_mat

        for layer in self.layers:
            row_emb, col_emb = layer(row_emb, col_emb, cost_mat)

        return row_emb, col_emb


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.row_encoding_block = EncodingBlock(**model_params)
        self.col_encoding_block = EncodingBlock(**model_params)

    def forward(self, row_emb, col_emb, cost_mat):
        # row_emb.shape: (batch, row_cnt, embedding)
        # col_emb.shape: (batch, col_cnt, embedding)
        # cost_mat.shape: (batch, row_cnt, col_cnt)
        row_emb_out = self.row_encoding_block(row_emb, col_emb, cost_mat)
        col_emb_out = self.col_encoding_block(col_emb, row_emb, cost_mat.transpose(1, 2))

        return row_emb_out, col_emb_out


class EncodingBlock(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']

        self.Wq = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.add_n_normalization_1 = AddAndInstanceNormalization(**model_params)
        self.feed_forward = FeedForward(**model_params)
        self.add_n_normalization_2 = AddAndInstanceNormalization(**model_params)

        self.alpha = nn.Parameter(torch.Tensor([1.]), requires_grad=True)

    def forward(self, row_emb, col_emb, cost_mat):
        # NOTE: row and col can be exchanged, if cost_mat.transpose(1,2) is used
        # input1.shape: (batch, row_cnt, embedding)
        # input2.shape: (batch, col_cnt, embedding)
        # cost_mat.shape: (batch, row_cnt, col_cnt)

        q = self.Wq(row_emb)
        # shape: (batch, row_cnt, embedding)
        k = self.Wk(col_emb)
        v = self.Wv(col_emb)
        # shape: (batch, col_cnt, embedding)

        alpha_relation_bias = self.alpha * cost_mat
        out_aft = adaptation_attention_free_module(q, k, v, adaptation_bias=alpha_relation_bias)
        # shape: (batch, row_cnt, embedding)

        out1 = self.add_n_normalization_1(row_emb, out_aft)
        out2 = self.feed_forward(out1)
        out3 = self.add_n_normalization_2(out1, out2)

        return out3
        # shape: (batch, row_cnt, embedding)


########################################
# Decoder
########################################

class ATSP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']

        self.Wq_0 = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wq_1 = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, embedding_dim, bias=False)


        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved key, for single-head attention
        self.q1 = None  # saved q1, for multi-head attention

        self.alpha1 = nn.Parameter(torch.Tensor([1.]), requires_grad=True)
        self.alpha2 = nn.Parameter(torch.Tensor([1.]), requires_grad=True)

    def set_kv(self, encoded_jobs):
        # encoded_jobs.shape: (batch, job, embedding)

        self.k = self.Wk(encoded_jobs)
        self.v = self.Wv(encoded_jobs)
        # shape: (batch, job, embedding)
        self.single_head_key = encoded_jobs.transpose(1, 2)
        # shape: (batch, embedding, job)

    def set_q1(self, encoded_q1):
        # encoded_q.shape: (batch, n, embedding)  # n can be 1 or pomo

        self.q1 = self.Wq_1(encoded_q1)
        # shape: (batch, n, embedding)

    def forward(self, encoded_q0, cur_cost_mat, log_scale, ninf_mask):
        # encoded_q4.shape: (batch, pomo, embedding)
        # ninf_mask.shape: (batch, pomo, job)

        #  We use adaptation_attention_free_module to replace multi-head attention
        #######################################################
        q0 = self.Wq_0(encoded_q0)
        # shape: (batch, pomo, embedding)

        q = self.q1 + q0
        # shape: (batch, pomo, embedding)

        #  We use AAFM to replace the multi-head attention
        #######################################################
        alpha_relation_bias = -1 * self.alpha1 * log_scale * cur_cost_mat
        AAFM_OUT = adaptation_attention_free_module(q, self.k, self.v,
                                                    adaptation_bias=alpha_relation_bias,
                                                    ninf_mask=ninf_mask)
        # shape: (batch, pomo, embedding)

        #  Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(AAFM_OUT, self.single_head_key)
        # shape: (batch, pomo, job)

        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        logit_clipping = self.model_params['logit_clipping']

        score_scaled = score / sqrt_embedding_dim
        # shape: (batch, pomo, job)

        score_scaled = score_scaled - self.alpha2 * log_scale * cur_cost_mat
        score_clipped = logit_clipping * torch.tanh(score_scaled)

        score_masked = score_clipped + ninf_mask

        probs = F.softmax(score_masked, dim=-1)
        # shape: (batch, pomo, job)

        return probs

########################################
# NN SUB CLASS / FUNCTIONS
########################################

def adaptation_attention_free_module(q, k, v, adaptation_bias, ninf_mask=None):
    """
    The core code of Adaptation Attention Free Module.

    Inspired by the paper: An Attention Free Transformer
    (url:  https://arxiv.org/pdf/2105.14103.pdf)

    Args:
        q: query, shape: (batch, n, embedding_dim)
        k: key, shape: (batch, m, embedding_dim)
        v: value, shape: (batch, m, embedding_dim)
        adaptation_bias: - alpha * log_scale * dist, shape: (batch, n, m)
        ninf_mask: shape: (batch, n, m)

    Return:
        out: shape: (batch, n, embedding_dim)

    Note:
    To prevent potential value overflows caused by exponential operations, we use "torch.nan_to_num" to solve it.
    For more details, please refer to the official document:
    https://pytorch.org/docs/1.10/generated/torch.nan_to_num.html
    """

    sigmoid_q = torch.sigmoid(q)
    # shape: (batch, n, embedding_dim)

    if ninf_mask is not None:
        adaptation_bias = adaptation_bias + ninf_mask

    bias = torch.exp(adaptation_bias) @ torch.mul(torch.exp(k), v)
    # shape: (batch, n, embedding_dim)
    a_k = torch.exp(adaptation_bias) @ torch.exp(k)

    weighted = bias / a_k
    if torch.isinf(bias).any() or torch.isinf(a_k).any():
        weighted = torch.nan_to_num_(bias) / torch.nan_to_num_(a_k)
    if torch.isnan(weighted).any():
        torch.nan_to_num_(weighted)
    # shape: (batch, n, embedding_dim)

    out = torch.mul(sigmoid_q, weighted)
    # shape: (batch, n, embedding_dim)

    return out


class AddAndInstanceNormalization(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)

    def forward(self, input1, input2):
        # input.shape: (batch, problem, embedding)

        added = input1 + input2
        # shape: (batch, problem, embedding)

        transposed = added.transpose(1, 2)
        # shape: (batch, embedding, problem)

        normalized = self.norm(transposed)
        # shape: (batch, embedding, problem)

        back_trans = normalized.transpose(1, 2)
        # shape: (batch, problem, embedding)

        return back_trans


class FeedForward(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        # input.shape: (batch, problem, embedding)

        return self.W2(F.relu(self.W1(input1)))


