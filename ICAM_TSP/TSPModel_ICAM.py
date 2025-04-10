
import torch
import torch.nn as nn
import torch.nn.functional as F


class TSPModel(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params

        self.encoder = TSP_Encoder(**model_params)
        self.decoder = TSP_Decoder(**model_params)
        self.encoded_nodes = None
        # shape: (batch, problem, embedding_dim)
        self.log_scale = None

    def set_decoder_type(self,decoder_type):
        self.model_params['eval_type'] = decoder_type

    def pre_forward(self, reset_state):

        self.log_scale = reset_state.log_scale # it is a scalar and used for influence of distance
        self.encoded_nodes = self.encoder(reset_state.problems,reset_state.dist,self.log_scale)
        # shape: (batch, problem, embedding_dim)
        self.decoder.set_kv(self.encoded_nodes)

    def forward(self, state,cur_dist):
        batch_size = state.batch_size
        pomo_size = state.pomo_size

        if state.current_node is None:
            selected = torch.arange(pomo_size)[None, :].expand(batch_size, pomo_size)
            prob = torch.ones(size=(batch_size, pomo_size))

            encoded_first_node = _get_encoding(self.encoded_nodes, selected)
            # shape: (batch, pomo, embedding)
            self.decoder.set_q1(encoded_first_node)

        else:
            encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
            # shape: (batch, pomo, embedding)
            probs = self.decoder(encoded_last_node, cur_dist, self.log_scale,ninf_mask=state.ninf_mask)
            # shape: (batch, pomo, problem)
            assert not torch.isnan(probs).any(), "probs has nan, but it should not have any nans."

            if self.training or self.model_params['eval_type'] == 'sampling':
                # Check if sampling went OK, can go wrong due to bug on GPU
                # See https://discuss.pytorch.org/t/bad-behavior-of-multinomial-function/10232
                # to fix pytorch.multinomial bug on selecting 0 probability elements
                while True:
                    selected = (probs.reshape(batch_size * pomo_size, -1).multinomial(1)
                                .squeeze(dim=1).reshape(batch_size, pomo_size))
                    # shape: (batch, pomo)

                    prob = torch.gather(probs, dim=-1, index=selected.unsqueeze(-1)).squeeze(dim=-1)
                    # shape: (batch, pomo)

                    if (prob != 0).all():
                        break

            elif self.model_params['eval_type'] == 'greedy':
                selected = probs.argmax(dim=-1)
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

class TSP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        encoder_layer_num = self.model_params['encoder_layer_num']

        self.embedding = nn.Linear(2, embedding_dim)
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])

    def forward(self, data,dist,log_scale):
        # data.shape: (batch, problem, 2)

        embedded_input = self.embedding(data)
        # shape: (batch, problem, embedding)

        out = embedded_input
        negative_scale_dist = -1 * log_scale * dist
        for layer in self.layers:
            out = layer(out,negative_scale_dist)

        return out


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']

        self.Wq = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.addAndNormalization1 = Add_And_Normalization_Module(**model_params)
        self.feedForward = Feed_Forward_Module(**model_params)
        self.addAndNormalization2 = Add_And_Normalization_Module(**model_params)

        self.alpha = nn.Parameter(torch.Tensor([1.]), requires_grad=True)

    def forward(self, input1,negative_scale_dist):
        # input.shape: (batch, problem, embedding_dim)
        # dist.shape: (batch, problem, problem)
        # scale.shape: (1,)

        q = self.Wq(input1)
        k = self.Wk(input1)
        v = self.Wv(input1)
        # shape: (batch, problem, embedding_dim)

        #  We use AAFM to replace the multi-head attention
        #######################################################
        alpha_dist_bias_scale = self.alpha * negative_scale_dist
        # shape: (batch, problem, problem)
        AAFM_OUT = adaptation_attention_free_module(q, k, v, alpha_dist_bias_scale)
        # shape: (batch, problem, embedding)

        out1 = self.addAndNormalization1(input1, AAFM_OUT)
        out2 = self.feedForward(out1)
        out3 = self.addAndNormalization2(out1, out2)

        return out3
        # shape: (batch, problem, EMBEDDING_DIM)


########################################
# DECODER
########################################

class TSP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']

        self.Wq_first = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wq_last = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved, for single-head attention
        self.q_first = None  # saved q1, for multi-head attention

        self.alpha1 = nn.Parameter(torch.Tensor([1.]), requires_grad=True)
        self.alpha2 = nn.Parameter(torch.Tensor([1.]), requires_grad=True)

    def set_kv(self, encoded_nodes):
        # encoded_nodes.shape: (batch, problem, embedding)

        self.k = self.Wk(encoded_nodes)
        self.v = self.Wv(encoded_nodes)
        # shape: (batch, problem, embedding)
        self.single_head_key = encoded_nodes.transpose(1, 2)
        # shape: (batch, embedding, problem)

    def set_q1(self, encoded_q1):
        # encoded_q.shape: (batch, n, embedding)  # n can be 1 or pomo
        self.q_first = self.Wq_first(encoded_q1)
        # shape: (batch, problem, embedding)

    def forward(self, encoded_last_node, cur_dist,log_scale,ninf_mask):
        # encoded_last_node.shape: (batch, pomo, embedding)
        # ninf_mask.shape: (batch, pomo, problem)
        # cur_dist.shape: (batch, pomo, problem)

        q_last = self.Wq_last(encoded_last_node)
        # shape: (batch, pomo, embedding_dim)
        q = self.q_first + q_last
        # shape: (batch, pomo, embedding_dim)

        #  We use AAFM to replace the multi-head attention
        #######################################################
        alpha_adaptation_bias = -1 * self.alpha1 * log_scale * cur_dist
        # shape: (batch, pomo, problem)
        AAFM_OUT = adaptation_attention_free_module(q, self.k, self.v, alpha_adaptation_bias, ninf_mask)
        # shape: (batch, pomo, embedding)

        #  Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(AAFM_OUT, self.single_head_key)
        # shape: (batch, pomo, problem)
        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        score_scaled = score / sqrt_embedding_dim
        # shape: (batch, pomo, problem)
        score_scaled = score_scaled - self.alpha2 * log_scale * cur_dist
        # shape: (batch, pomo, problem)
        logit_clipping = self.model_params['logit_clipping']
        score_clipped = logit_clipping * torch.tanh(score_scaled)
        # shape: (batch, pomo, problem)
        score_masked = score_clipped + ninf_mask

        probs = F.softmax(score_masked, dim=-1)
        # shape: (batch, pomo, problem)

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

class Add_And_Normalization_Module(nn.Module):
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


class Feed_Forward_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        # input.shape: (batch, problem, embedding)

        return self.W2(F.relu(self.W1(input1)))
