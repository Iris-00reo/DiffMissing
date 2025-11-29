import torch.utils.data
from diff_block import DenoisingNetwork
from TimeMixer import TimeMixer
import torch
from torch import nn


class DiffMiss(nn.Module):
    def __init__(self, config, device):
        super(DiffMiss, self).__init__()
        self.device = device
        self.node_num = config["model"]["node_num"]
        self.missing_ratio = config["model"]["missing_ratio"]
        self.seq_len = config["model"]["seq_len"]
        self.config_diff = config['diffusion']

        self.denoising = DenoisingNetwork(self.config_diff, self.node_num, self.missing_ratio, self.seq_len, self.device).to(self.device)

        # parameters for diffusion models
        self.num_steps = self.config_diff["num_steps"]
        if self.config_diff["schedule"] == "quad":
            self.beta = torch.linspace(self.config_diff["beta_start"] ** 0.5, self.config_diff["beta_end"] ** 0.5, self.num_steps,
                                       device=self.device)**2
        elif self.config_diff["schedule"] == "linear":
            self.beta = torch.linspace(self.config_diff["beta_start"], self.config_diff["beta_end"], self.num_steps,
                                       device=self.device)

        self.alpha_hat = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha_hat, dim=0)

        self.sigma2 = torch.cat((torch.tensor([self.beta[0]], device=self.device),
                                 self.beta[1:]*(1-self.alpha_bar[0:-1])/(1-self.alpha_bar[1:])))

        self.loss_func = nn.MSELoss()
        # self.output_linear = nn.Linear(in_features=self.config_diff["in_size"], out_features=1, bias=True)
        self.output_linear = nn.Linear(in_features=self.config_diff["in_size"], out_features=self.config_diff["in_size"], bias=True)

        self.timemixer = TimeMixer(configs=config)
        self.output = nn.Linear(in_features=self.config_diff["in_size"] * self.node_num, out_features=self.node_num, bias=True)


    def gather(self, const, t):
        # return const.gather(-1, t).contiguous().view(-1, 1, 1, 1)
        return const.gather(-1, t).reshape(-1, 1, 1, 1)

    def q_xt_x0(self, x0, t):
        alpha_bar = self.gather(self.alpha_bar, t)
        mean = (alpha_bar**0.5)*x0
        var = 1 - alpha_bar
        return mean, var

    def q_sample(self, x0, t, eps):
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var**0.5)*eps

    def p_sample(self, xt, adj_mx, mask_id, is_train, t):
        eps_theta = self.denoising(xt, adj_mx, mask_id, is_train, t)
        alpha_bar = self.gather(self.alpha_bar, t)
        alpha = self.gather(self.alpha_hat, t)
        eps_coef = (1 - alpha)/(1 - alpha_bar)**0.5
        mean = (xt - eps_coef*eps_theta)/(alpha**0.5)
        var = self.gather(self.sigma2, t)
        if (t == 0).all():
            z = torch.zeros(xt.shape, device=xt.device)
        else:
            z = torch.randn(xt.shape, device=xt.device)
        return mean + (var**0.5)*z

    def cal_loss(self, x0, adj_mx, mask_id, is_train):
        batch_size = x0.shape[0]

        # add noise to observed data
        t = torch.randint(0, self.num_steps, (batch_size,), device=self.device, dtype=torch.long)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, eps=noise)

        eps_theta = self.denoising(xt, adj_mx, mask_id, is_train, t)    # B, C, L, N
        loss_noise = self.loss_func(noise[:, :, :, mask_id], eps_theta[:, :, :, mask_id])
        B, C, L, N = eps_theta.shape
        # output = self.output_linear(eps_theta.reshape(B, -1, L*N).transpose(-1, -2)).transpose(-1, -2).reshape(B, L, N)
        output = self.output_linear(eps_theta.reshape(B, -1, L*N).transpose(-1, -2)).transpose(-1, -2).reshape(B, L, N, C)

        return loss_noise, output

    def forward(self, x0, adj_mx=None, mask_id=None, is_train=1):
        # Input [B, L, N, C], Output [B, Co, L, N], H == L, Co= C or 1
        x0 = x0.transpose(-1, -3).transpose(-1, -2)     # B, C, N, H ==> B, C, H, N

        return self.cal_loss(x0, adj_mx, mask_id, is_train)

    def forecast(self, x0, xt, mask_id=None, is_train=1):
        B, L, N, C = x0.shape
        anti_mask_id = torch.arange(N, device=mask_id.device)[~torch.isin(torch.arange(N, device=mask_id.device), mask_id)]
        anti_mask = torch.zeros(B, L, N, C, dtype=torch.bool, device=xt.device)
        anti_mask[:, :, anti_mask_id, :] = True
        xt = xt.masked_fill(anti_mask, 0.0)
        x = x0 + xt
        x = x.reshape(B, -1, N*C)
        output = self.timemixer(x, None, None, None, is_train)  # B L Co
        output = self.output(output)
        return output

    def inference(self, x, n_samples, adj_mx=None, mask_id=None, is_train=0, batch_idx=0):
        B = x.size(0)
        pred_samples = torch.zeros(B, n_samples, self.seq_len, self.node_num).to(self.device)

        # if n_samples != 1:
        #     for i in range(n_samples):
        #         xt = torch.randn([B, self.seq_len, self.node_num]).to(self.device).unsqueeze(1)
        #         for j in range(self.num_steps - 1, -1, -1):
        #             t = (torch.ones(B) * j).long().to(self.device)
        #             xt = self.p_sample(xt, adj_mx, mask_id, is_train=is_train, t=t)
        #
        #         pred_samples[:, i] = xt.squeeze(1).detach()
        # else:
        #     num_steps = self.num_steps // 10
        #     idx = 9 - (batch_idx-1) % 10
        #     xt = torch.randn([B, self.seq_len, self.node_num]).to(self.device).unsqueeze(1)
        #     for j in range(num_steps*(idx+1) - 1, num_steps*idx - 1, -1):
        #         t = (torch.ones(B) * j).long().to(self.device)
        #         xt = self.p_sample(xt, adj_mx, mask_id, is_train=is_train, t=t)
        #
        #     pred_samples[:, 0] = xt.squeeze(1).detach()

        for i in range(n_samples):
            num_steps = self.num_steps // 10
            idx = 9 - (batch_idx-1) % 10
            xt = torch.randn([B, self.seq_len, self.node_num]).to(self.device).unsqueeze(1)
            for j in range(num_steps*(idx+1) - 1, num_steps*idx - 1, -1):
                t = (torch.ones(B) * j).long().to(self.device)
                xt = self.p_sample(xt, adj_mx, mask_id, is_train=is_train, t=t)

            pred_samples[:, i] = xt.squeeze(1).detach()
        return pred_samples

