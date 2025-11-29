from torch import nn
import torch
import torch.nn.functional as F
import math


def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu")
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)


def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class SelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, hq, hv, hk, dropout):
        super(SelfAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scaling = float(self.head_dim)**-0.5
        self.q_proj_layer = nn.Linear(hq, d_model, bias=True)
        self.k_proj_layer = nn.Linear(hk, d_model, bias=True)
        self.v_proj_layer = nn.Linear(hv, d_model, bias=True)
        self.norm = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model, bias=True),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(d_model, d_model, bias=True),
            nn.Dropout(p=dropout),
            nn.ReLU()
        )

    def forward(self, query, key, value):
        # query: (B, L, d_cross)
        Q = self.q_proj_layer(query).transpose(0, 1)    # (L, B, d_model)
        K = self.k_proj_layer(key).transpose(0, 1)
        V = self.v_proj_layer(value).transpose(0, 1)

        seq_len, batch_size = Q.shape[0], Q.shape[1]
        Q1 = Q.reshape(seq_len, batch_size*self.num_heads, self.head_dim).transpose(0, 1)    # (B*num_heads, L, head_dim)
        K1 = K.reshape(seq_len, batch_size*self.num_heads, self.head_dim).transpose(0, 1)
        V1 = V.reshape(seq_len, batch_size*self.num_heads, self.head_dim).transpose(0, 1)

        att_weights = torch.bmm(Q1, K1.transpose(1, 2))*self.scaling      # (B*num_heads, L, L)
        att_weights = F.softmax(att_weights, dim=-1)

        att_output0 = torch.bmm(att_weights, V1)     # (B*num_heads, L, head_dim)
        att_output0 = att_output0.transpose(0, 1).reshape(seq_len, batch_size, self.d_model).transpose(0, 1)    # (B, L, d_model)

        attn_output1 = att_output0 + K.transpose(-2, -3)
        attn_output1 = self.norm(attn_output1)
        attn_output2 = self.feed_forward(attn_output1)
        attn_output3 = attn_output2 + attn_output1
        attn_output3 = self.norm(attn_output3)
        return attn_output3


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, bidirectional=True):
        super(BiLSTM, self).__init__()

        # BiLSTM Layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, bidirectional=bidirectional, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, output_size)

    def forward(self, x):
        # x: (batch_size, input_size, seq_length)
        lstm_out, (h_n, c_n) = self.lstm(x.transpose(-1, -2))
        # use the hidden states of both directions for the last time step
        if self.lstm.bidirectional:
            y = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            y = h_n[-1]
        y = torch.unsqueeze(y, dim=-1)
        z = lstm_out.transpose(-1, -2) + y
        output = self.fc(z.transpose(-1, -2)).transpose(-1, -2)
        return output


class CrossAttention(nn.Module):
    def __init__(self, d_cross, num_heads, d_local, d_global, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.att = SelfAttention(d_model=d_cross, num_heads=num_heads, hq=d_local,
                                 hv=d_global, hk=d_global, dropout=dropout)

    def forward(self, local_info, global_info):
        # local_info: (B,  C1, L)
        att_enc = self.att(local_info.transpose(-1, -2), global_info.transpose(-1, -2), global_info.transpose(-1, -2))
        att_enc = att_enc.transpose(-1, -2)     # B, d_cross, L
        return att_enc


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super(DiffusionEmbedding, self).__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer("embedding", self._build_embedding(num_steps, embedding_dim // 2), persistent=False)
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table


class ConditionNetwork(nn.Module):
    def __init__(self, channels, d_cross, num_heads, dropout, node_num, missing_ratio, device):
        super(ConditionNetwork, self).__init__()
        self.device = device

        self.conv1 = nn.Sequential(
            Conv1d_with_init(channels, channels*2, kernel_size=1),
            nn.ReLU()
        )
        self.bilstm = BiLSTM(input_size=channels, hidden_size=channels*4, output_size=channels*2)
        self.cross_att = CrossAttention(d_cross=d_cross, num_heads=num_heads, d_local=channels*2, d_global=channels*2, dropout=dropout)
        self.linear = nn.Sequential(
            nn.Linear(in_features=node_num - round(node_num*missing_ratio), out_features=node_num),
            nn.SiLU()
        )

    def forward(self, x, mask_id):
        B, C, H, N = x.shape

        full_index = torch.arange(x.size(-1)).to(self.device)
        normal_id = full_index[~torch.isin(full_index, mask_id)]
        x_norm = x[:, :, :, normal_id]

        x_norm = x_norm.reshape(B, C, H * len(normal_id))
        local_info = self.conv1(x_norm)     # (B, C1, H * N_norm)
        global_info = self.bilstm(x_norm)   # (B, C1, H * N_norm)

        y = self.cross_att(local_info, global_info)     # B, d_cross, H * N_norm
        y = y.reshape(B, -1, H, len(normal_id))   # B, d_cross, H, N_normal
        c = self.linear(y)      # B, Cx, L, N
        return c


class SparseAttention(nn.Module):
    def __init__(self, in_c, out_c, node_num, grap_size, dropout, K):
        super(SparseAttention, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.node_num = node_num
        self.top_k = int(self.node_num * K)
        self.dropout = nn.Dropout(dropout)

        self.W = nn.Parameter(torch.FloatTensor(size=(self.in_c, self.out_c)))
        nn.init.xavier_uniform_(self.W)  # initialize
        self.a = nn.Parameter(torch.FloatTensor(size=(2 * self.out_c, 1)))
        nn.init.xavier_uniform_(self.a)  # initialize

        self.LR = nn.LeakyReLU()
        self.GL = nn.Parameter(torch.FloatTensor(self.node_num, grap_size))
        nn.init.kaiming_uniform_(self.GL)

    def forward(self, x):
        # x: input_fea [B, N, C*L]
        B, N = x.size(0), x.size(1)
        adj = F.softmax(F.relu(self.GL @ self.GL.transpose(-2, -1)), dim=-1)
        adj = adj + torch.eye(N, dtype=adj.dtype, device=adj.device)
        h = torch.matmul(x, self.W)  # [B, N, out_c]

        from torch.amp import autocast
        with autocast('cuda'):
            torch.cuda.empty_cache()
            # a_input = torch.cat([h.repeat(1, 1, N).contiguous().view(B, N * N, -1), h.repeat(1, N, 1)], dim=2).contiguous().view(B, N, -1, 2 * self.out_c)    # [B, N, N, 2 * out_c]
            a_input = torch.cat([h.repeat(1, 1, N).reshape(B, N * N, -1), h.repeat(1, N, 1)], dim=2).reshape(B, N, -1, 2 * self.out_c)    # [B, N, N, 2 * out_c]

        e = self.LR(torch.matmul(a_input, self.a).squeeze(3))    # [B, N, N]
        zero_vec = -1e12 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)  # [B, N, N]
        top_k_values, top_k_indices = torch.topk(attention, self.top_k, dim=1)          # select node of top_k
        mask_top_k = torch.zeros_like(attention).scatter(1, top_k_indices, 1)   # construct sparsify mask
        attention = attention * mask_top_k      # sparsify attention weight

        attention = self.dropout(F.softmax(attention, dim=2))   # [N, N]
        h_prime = F.relu(torch.matmul(attention, self.dropout(h)))  # [B, N, N].[B, N, out_features] => [B,N, out_features]
        return h_prime.transpose(-2, -1)


class ResidualBlock(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, num_heads, node_num,
                 seq_len, grap_size, dropout, K=0.2):
        super(ResidualBlock, self).__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(side_dim, channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.spare_attention = SparseAttention(in_c=channels*seq_len, out_c=channels*seq_len, node_num=node_num,
                                               grap_size=grap_size, dropout=dropout, K=K)

        self.lstm_layer = nn.LSTM(input_size=channels, hidden_size=grap_size, num_layers=1, batch_first=True)
        self.linear = nn.Sequential(
            nn.Linear(in_features=grap_size, out_features=channels, bias=True),
            nn.SiLU())
        self.iTransformer_layer = get_torch_trans(heads=num_heads, layers=1, channels=channels)

    def forward_time(self, y, base_shape):
        B, channel, L, N = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, N, L).permute(0, 2, 1, 3).reshape(B * N, channel, L)
        y, (_, _) = self.lstm_layer(y.permute(2, 0, 1))         # (L, B*N, C)
        y = self.linear(y).permute(1, 2, 0)     # (B*N, C, L)
        y = y.reshape(B, N, channel, L).permute(0, 2, 1, 3).reshape(B, channel, N * L)
        return y

    def forward_feature(self, y, base_shape):
        B, channel, L, N = base_shape
        if N == 1:
            return y
        y = y.reshape(B, channel, N, L).permute(0, 3, 1, 2).reshape(B * L, channel, N)
        y = self.iTransformer_layer(y.permute(2, 0, 1)).permute(1, 2, 0)  # (N, B*L, C) ==> (B*L, C, N)
        y = y.reshape(B, L, channel, N).permute(0, 2, 3, 1).reshape(B, channel, N * L)
        return y

    def forward(self, x, cond_info, diffusion_emb, mask_id):
        B, channel, L, N = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, L * N)

        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  # (B,channel,1)
        y = x + diffusion_emb  # (B, channel, L*N)

        _, cond_dim, _, _ = cond_info.shape
        cond_info = cond_info.reshape(B, cond_dim, L * N)
        cond_info = self.cond_projection(cond_info)  # (B, channel, L*N)
        y = y + cond_info.reshape(B, channel, L * N)

        y = self.spare_attention(y.reshape(B, channel, L, N).transpose(-1, -3).reshape(B, N, -1))
        y = self.forward_time(y, base_shape)     # (B, channel, L*N)
        y = self.forward_feature(y, base_shape)  # (B, channel, L*N)
        y = self.mid_projection(y)  # (B, 2*channel, L*N)

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B, channel, N*L)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)  # (B, channel, L, N)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)  # (B, channel, L, N)
        return (x + residual) / math.sqrt(2.0), skip


class Encoder(nn.Module):
    def __init__(self, layers):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x, cond_info, diffusion_emb, mask_id):
        skips = []
        for layer in self.layers:
            x, skip = layer(x, cond_info, diffusion_emb, mask_id)
            skips.append(skip)
        skip_concat = torch.sum(torch.stack(skips), dim=0) / math.sqrt(len(self.layers))
        B, _, K, L = skip_concat.shape
        return skip_concat.reshape(B, -1, K * L)

class Decoder(nn.Module):
    def __init__(self, channels, output_dim):
        super(Decoder, self).__init__()
        self.output_projection1 = Conv1d_with_init(channels, channels, 1)
        self.output_projection2 = Conv1d_with_init(channels, output_dim, 1)
        nn.init.zeros_(self.output_projection2.weight)

    def forward(self, x_hidden, B, N, L):  # (B, channel, N*L) => (B, L, N)
        x = F.relu(self.output_projection1(x_hidden))      # (B, channel, N*L)
        x = self.output_projection2(x)  # (B, 1, N*L)
        x = x.reshape(B, -1, L, N)
        return x

class DenoisingNetwork(nn.Module):
    def __init__(self, config, node_num, missing_ratio, seq_len, device):
        super(DenoisingNetwork, self).__init__()

        self.channels = config["channels"]
        self.d_cross = config["d_cross"]
        self.num_heads = config["num_heads"]
        self.dropout = config["dropout"]
        self.input_dim = config["in_size"]
        self.output_dim = self.input_dim
        self.node_num = node_num
        self.missing_ratio = missing_ratio
        self.seq_len = seq_len
        config["side_dim"] = self.d_cross

        self.input_projection = Conv1d_with_init(self.input_dim, self.channels, kernel_size=1)
        self.infer_emb = Conv1d_with_init(in_channels=1, out_channels=self.channels, kernel_size=1)
        self.diffusion_embedding = DiffusionEmbedding(num_steps=config["num_steps"], embedding_dim=config["diffusion_embedding_dim"])

        self.cond_network = ConditionNetwork(self.channels, self.d_cross, self.num_heads, self.dropout, self.node_num, self.missing_ratio, device)
        self.cond_linear = nn.Linear(in_features=self.channels, out_features=self.d_cross)
        self.encoder = Encoder(nn.ModuleList(
            [ResidualBlock(side_dim=config["side_dim"], channels=self.channels,
                           diffusion_embedding_dim=config["diffusion_embedding_dim"], num_heads=self.num_heads,
                           node_num=node_num, seq_len=self.seq_len, grap_size=config["grap_size"],
                           dropout=self.dropout, K=config["K"]) for _ in range(config["layers"])]))

        self.decoder = Decoder(self.channels, self.output_dim)

        self.output_projection = nn.Sequential(
            nn.Linear(in_features=self.channels, out_features=self.channels, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=self.channels, out_features=self.output_dim, bias=True)
        )
        self.infer_projection = nn.Sequential(
            nn.Linear(in_features=self.channels, out_features=self.channels, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=self.channels, out_features=1, bias=True)
        )

    def __embedding(self, x, is_train):
        B, input_dim, L, N = x.shape
        if x is None:  # for pred_x in validation phase
            return None
        x = x.reshape(B, input_dim, N * L)
        if is_train == 1:
            x = F.relu(self.input_projection(x))
        else:
            x = F.relu(self.infer_emb(x))
        x = x.reshape(B, self.channels, L, N)
        return x


    def forward(self, x, adj_mx, mask_id, is_train, diffusion_step):
        B, input_dim, L, N = x.shape

        x_hidden = self.__embedding(x, is_train)      # B, C, L, N
        diffusion_emb = self.diffusion_embedding(diffusion_step)    # B, C1
        if is_train == 1:
            cond_info = self.cond_network(x_hidden, mask_id)    # B, Cx, L, N
        else:
            cond_info = self.cond_linear(x_hidden.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)

        forward_noise_hidden = self.encoder(x_hidden, cond_info, diffusion_emb, mask_id)    # (B, C, N*L)

        if is_train == 1:
            forward_noise = self.output_projection(forward_noise_hidden.transpose(-1, -2)).transpose(-1, -2).reshape(B, -1, L, N)
        else:
            forward_noise = self.infer_projection(forward_noise_hidden.transpose(-1, -2)).transpose(-1, -2).reshape(B, -1, L, N)
        return forward_noise
