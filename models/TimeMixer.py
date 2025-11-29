import torch
import torch.nn as nn
import math


class Normalize(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=False, non_norm=False):
        super(Normalize, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.non_norm = non_norm
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        return x

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.non_norm:
            return x
        x = (x - self.mean) / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, kernel_size=3,
                                   padding=padding, padding_mode='circular', bias=False)
        nn.init.kaiming_normal_(self.tokenConv.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        return self.tokenConv(x.permute(0, 2, 1).contiguous()).transpose(1, 2)


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()
        w = torch.zeros(c_in, d_model).float()
        w.requires_grad = False
        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)
        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()
        minute_size, hour_size, weekday_size, day_size, month_size = 4, 24, 7, 32, 13
        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        self.minute_embed = Embed(minute_size, d_model) if freq == 't' else None
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        return self.hour_embed(x[:, :, 3]) + self.weekday_embed(x[:, :, 2]) + \
            self.day_embed(x[:, :, 1]) + self.month_embed(x[:, :, 0]) + minute_x


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()
        self.value_embedding = TokenEmbedding(c_in, d_model)
        self.temporal_embedding = TemporalEmbedding(d_model, embed_type, freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x is None:
            return self.temporal_embedding(x_mark)
        x = self.value_embedding(x) + (self.temporal_embedding(x_mark) if x_mark is not None else 0)
        return self.dropout(x)


class moving_avg(nn.Module):
    def __init__(self, kernel_size):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size, stride=1, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        return self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)


class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        return x - moving_mean, moving_mean


class MultiScaleSeasonMixing(nn.Module):
    def __init__(self, configs):
        super(MultiScaleSeasonMixing, self).__init__()
        self.seq_len = configs["model"]['seq_len']
        self.window = configs["timemixer"]['down_sampling_window']
        self.layers = configs["timemixer"]['down_sampling_layers']

        self.down_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.seq_len // (self.window ** i), self.seq_len // (self.window ** (i + 1))),
                nn.GELU()
            ) for i in range(self.layers)
        ])

    def forward(self, season_list):
        out_high = season_list[0]
        out_low = season_list[1]
        out_list = [out_high.permute(0, 2, 1)]

        for i in range(len(season_list) - 1):
            out_low = out_low + self.down_layers[i](out_high)
            out_high = out_low
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]
            out_list.append(out_high.permute(0, 2, 1))
        return out_list


class MultiScaleTrendMixing(nn.Module):
    def __init__(self, configs):
        super(MultiScaleTrendMixing, self).__init__()
        self.seq_len = configs["model"]['seq_len']
        self.window = configs["timemixer"]['down_sampling_window']
        self.layers = configs["timemixer"]['down_sampling_layers']

        self.up_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.seq_len // (self.window ** (i + 1)), self.seq_len // (self.window ** i)),
                nn.GELU()
            ) for i in reversed(range(self.layers))
        ])

    def forward(self, trend_list):
        trend_list_rev = trend_list[::-1]
        out_low, out_high = trend_list_rev[0], trend_list_rev[1]
        out_list = [out_low.permute(0, 2, 1)]

        for i in range(len(trend_list_rev) - 1):
            out_high = out_high + self.up_layers[i](out_low)
            out_low = out_high
            if i + 2 <= len(trend_list_rev) - 1:
                out_high = trend_list_rev[i + 2]
            out_list.append(out_low.permute(0, 2, 1))
        return out_list[::-1]


class PastDecomposableMixing(nn.Module):
    def __init__(self, configs):
        super(PastDecomposableMixing, self).__init__()
        self.decomp = series_decomp(configs["timemixer"]["moving_avg"])
        self.channel_independence = configs["timemixer"]["channel_independence"]
        self.d_model = configs["timemixer"]["d_model"]
        self.d_ff = configs["timemixer"]["d_ff"]

        if not self.channel_independence:
            self.cross_layer = nn.Sequential(
                nn.Linear(self.d_model, self.d_ff),
                nn.GELU(),
                nn.Linear(self.d_ff, self.d_model)
            )

        self.season_mixer = MultiScaleSeasonMixing(configs)
        self.trend_mixer = MultiScaleTrendMixing(configs)
        self.out_layer = nn.Linear(self.d_model, self.d_model)

    def forward(self, x_list):
        length_list = [x.size(1) for x in x_list]
        season_list, trend_list = [], []

        for x in x_list:
            season, trend = self.decomp(x)
            if not self.channel_independence:
                season, trend = self.cross_layer(season), self.cross_layer(trend)
            season_list.append(season.permute(0, 2, 1))
            trend_list.append(trend.permute(0, 2, 1))

        out_season = self.season_mixer(season_list)
        out_trend = self.trend_mixer(trend_list)

        return [ori + self.out_layer(s + t)[:, :length, :]
                for ori, s, t, length in zip(x_list, out_season, out_trend, length_list)]


class TimeMixer(nn.Module):
    def __init__(self, configs):
        super(TimeMixer, self).__init__()
        self.configs = configs
        self.seq_len = configs["model"]["seq_len"]
        self.window = configs["timemixer"]["down_sampling_window"]
        self.layers = configs["timemixer"]["down_sampling_layers"]
        self.channel_independence = configs["timemixer"]["channel_independence"]

        self.pdm_blocks = nn.ModuleList([PastDecomposableMixing(configs) for _ in range(configs["timemixer"]["e_layers"])])
        self.preprocess = series_decomp(configs["timemixer"]["moving_avg"])
        self.enc_in = configs["model"]["node_num"] * configs["timemixer"]["in_size"]
        self.c_out = self.enc_in
        self.d_model = configs["timemixer"]["d_model"]

        self.enc_embedding = DataEmbedding(self.enc_in, self.d_model, configs["timemixer"]["embed"],
                                           configs["timemixer"]["freq"], configs["timemixer"]["dropout"])

        self.norm_layers = nn.ModuleList([Normalize(self.enc_in, affine=True, non_norm=(configs["timemixer"]["use_norm"] == 0))
                                          for _ in range(self.layers + 1)])

        self.predict_layers = nn.ModuleList([nn.Linear(self.seq_len // (self.window ** i), self.seq_len)
                                             for i in range(self.layers + 1)])

        self.proj = nn.Linear(self.d_model, 1 if self.channel_independence else self.c_out)
        if not self.channel_independence:
            self.res_layers = nn.ModuleList([
                nn.Linear(self.seq_len // (self.window ** i), self.seq_len)
                for i in range(self.layers + 1)
            ])

    def out_projection(self, dec_out, i, out_res):
        dec_out = self.proj(dec_out)
        out_res = self.res_layers[i](out_res.permute(0, 2, 1)).permute(0, 2, 1)
        dec_out = dec_out + out_res
        return dec_out

    def pre_enc(self, x_list):
        return zip(*[self.preprocess(x) for x in x_list])

    def _multi_scale_process(self, x_enc, x_mark_enc):
        down_pool = nn.AvgPool1d(self.window)
        x_enc = x_enc.permute(0, 2, 1)  # B,T,C -> B,C,T
        x_enc_list = [x_enc.permute(0, 2, 1)]
        x_mark_list = [x_mark_enc] if x_mark_enc is not None else []

        x_enc_ori, x_mark_ori = x_enc, x_mark_enc
        for _ in range(self.layers):
            x_enc_ori = down_pool(x_enc_ori)
            x_enc_list.append(x_enc_ori.permute(0, 2, 1))

            if x_mark_ori is not None:
                x_mark_ori = x_mark_ori[:, ::self.window, :]
                x_mark_list.append(x_mark_ori)

        return x_enc_list, x_mark_list if x_mark_enc is not None else None

    def forecast(self, x_enc, x_mark_enc):
        x_enc, x_mark_enc = self._multi_scale_process(x_enc, x_mark_enc)

        x_list = []
        for i, x in enumerate(x_enc):
            x_list.append(self.norm_layers[i](x, 'norm'))

        # embedding
        x1_list, x2_list = self.pre_enc(x_list)
        enc_out_list = []
        if x_mark_enc is not None:
            for x, x_mark in zip(x1_list, x_mark_enc):
                enc_out_list.append(self.enc_embedding(x, x_mark))
        else:
            enc_out_list = [self.enc_embedding(x, None) for x in x1_list]

        # Past Decomposable Mixing as encoder for past
        for block in self.pdm_blocks:
            enc_out_list = block(enc_out_list)

        # Future Multipredictor Mixing as decoder for future
        dec_out_list = []
        for i, enc_out, out_res in zip(range(len(enc_out_list)), enc_out_list, x2_list):
            dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(0, 2, 1)
            dec_out_list.append(self.out_projection(dec_out, i, out_res))

        dec_out = torch.stack(dec_out_list, dim=-1).sum(-1)
        return self.norm_layers[0](dec_out, 'denorm')

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, is_train=0):
        return self.forecast(x_enc, x_mark_enc)

