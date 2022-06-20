import torch
import torch.nn as nn
import math

class DataEmbedding(nn.Module):
    def __init__(self, in_dim, out_dim, en_dim):
        super(DataEmbedding, self).__init__()
        
        self.linear_1 = nn.Linear(in_dim, out_dim)
        self.conv = nn.Conv1d(in_channels=in_dim, out_channels=out_dim, kernel_size=3, padding=1)
        self.activation = nn.ELU()
        self.linear_2 = nn.Linear(out_dim, en_dim)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.activation(x)
        x = self.linear_2(x)

        return x


class AutoCorrelation(nn.Module):
    def __init__(self, factor=1, dropout=0.1, n_head=4):
        super(AutoCorrelation, self).__init__()
        self.factor = factor
        self.dropout = nn.Dropout(dropout)
        self.n_head = n_head

    def time_delay_agg_training(self, values, corr):
        _, head, channel, length = values.shape

        # Find top k possible period tau
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]
        weights = torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)
        
        # Aggregation
        tmp_corr = torch.softmax(weights, dim=-1)
        tmp_values = values
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            pattern = torch.roll(tmp_values, -int(index[i]), -1)
            delays_agg = delays_agg + pattern * \
                         (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))

        return delays_agg

    def time_delay_agg_inference(self, values, corr):
        batch, head, channel, length = values.shape

        # index init
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch, head, channel, 1)

        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        weights = torch.topk(mean_value, top_k, dim=-1)[0]
        delay = torch.topk(mean_value, top_k, dim=-1)[1]
        
        # Aggregation
        tmp_corr = torch.softmax(weights, dim=-1)
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * \
                         (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
        
        return delays_agg


    def forward(self, queries, keys, values, attn_mask=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_head

        queries = queries.view(B, L, H, -1)
        keys = keys.view(B, S, H, -1)
        values = values.view(B, S, H, -1)

        if L > S:
            zeros = torch.zeros_like(queries[:, :(L - S), :]).float()
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]

        # Autocorrelation coefficient
        q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)
        spec = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(spec, dim=-1)

        V = self.time_delay_agg_training(values.permute(0, 2, 3, 1).contiguous(), corr) if self.training else \
            self.time_delay_agg_inference(values.permute(0, 2, 3, 1).contiguous(), corr)

        return V.permute(0, 3, 1, 2).view(B, L, -1), corr.permute(0, 3, 1, 2).view(B, L, -1)


class LayerNormAD(nn.Module):
    def __init__(self, channels):
        super(LayerNormAD, self).__init__()
        self.layer_norm = nn.LayerNorm(channels)

    def forward(self, x):
        x_hat = self.layer_norm(x)
        # Eliminate mu-shift
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)

        return x_hat - bias


class SeriesDecomp(nn.Module):
    def __init__(self, sliding_size, stride=1):
        super(SeriesDecomp, self).__init__()
        self.sliding_size = sliding_size
        self.avg = nn.AvgPool1d(kernel_size=sliding_size, stride=stride, padding=0)

    def forward(self, x):
        # Padding for alignment
        front = x[:, 0:1, :].repeat(1, (self.sliding_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.sliding_size - 1) // 2, 1)
        x_pad = torch.cat([front, x, end], dim=1)
        mean = self.avg(x_pad.permute(0, 2, 1)).permute(0, 2, 1)

        return x - mean, mean


class ConvFeedForward(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(ConvFeedForward, self).__init__()

        self.conv_1 = nn.Conv1d(in_channels=in_dim, out_channels=hidden_dim, kernel_size=1, bias=False)
        self.conv_2 = nn.Conv1d(in_channels=hidden_dim, out_channels=in_dim, kernel_size=1, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv_1(x.transpose(-1, 1))
        x = self.relu(x)
        x = self.conv_2(x)

        return x.transpose(-1, 1)


class Encoder(nn.Module):
    def __init__(self, d_model, d_ff=None, sliding_size=25, n_head=4):
        super(Encoder, self).__init__()

        self.attention = AutoCorrelation(n_head=n_head)
        self.ffn = ConvFeedForward(d_model, d_ff or 4 * d_model)
        self.decomp_1 = SeriesDecomp(sliding_size)
        self.decomp_2 = SeriesDecomp(sliding_size)
        self.norm = LayerNormAD(d_model)

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + new_x
        x, _ = self.decomp_1(x)

        res = x
        res = self.ffn(res)
        season, _ = self.decomp_2(x + res)
        season = self.norm(season)

        return season, attn


class Decoder(nn.Module):
    def __init__(self, d_model, c_out, d_ff=None, sliding_size=25, n_head=4):
        super(Decoder, self).__init__()

        self.attention = AutoCorrelation(n_head=n_head)
        self.ffn = ConvFeedForward(d_model, d_ff or 4 * d_model)
        self.decomp_1 = SeriesDecomp(sliding_size)
        self.decomp_2 = SeriesDecomp(sliding_size)
        self.decomp_3 = SeriesDecomp(sliding_size)
        self.norm = LayerNormAD(d_model)
        self.projection_seasonality = nn.Linear(d_model, c_out)
        self.projection_trend = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=3, stride=1, padding=1,
                                    padding_mode='circular', bias=False)

    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
        x = x + self.attention(
            x, x, x,
            attn_mask=x_mask
        )[0]
        x, trend_1 = self.decomp_1(x)

        x = x + self.attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0]
        x, trend_2 = self.decomp_2(x)
        
        res = x
        res = self.ffn(res)
        x, trend_3 = self.decomp_3(x + res)

        season = self.norm(x)
        season = self.projection_seasonality(season)

        residual_trend = trend_1 + trend_2 + trend_3
        residual_trend = self.projection_trend(residual_trend.permute(0, 2, 1)).transpose(1, 2)

        return season, trend + residual_trend


class Autoformer(nn.Module):
    def __init__(self, seq_len=96, pred_len=96, sliding_size=25, d_model=256, n_head=4, c_out=7):
        super(Autoformer, self).__init__()
        
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.decomp = SeriesDecomp(sliding_size)
        self.data_embed = DataEmbedding(c_out, c_out*4, d_model)
        self.encoder = Encoder(d_model=d_model, n_head=n_head)
        self.decoder = Decoder(d_model=d_model, c_out=c_out)

    def forward(self, x_enc, x_dec, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # Initialization
        trend_init = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        seasonal_init = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)
        
        # [B, seq_len, D]
        x_enc = self.data_embed(x_enc)
        enc_out, attns = self.encoder(x_enc, attn_mask=enc_self_mask)
        # [B, pred_len, D]
        dec_out = self.data_embed(seasonal_init)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, 
                                    cross_mask=dec_enc_mask, trend=trend_init)
        # Final prediction
        dec_out = trend_part + seasonal_part
        # [B, pred_len, D]
        return dec_out[:, -self.pred_len:, :], attns


if __name__ == '__main__':
    x = torch.rand((32, 96, 4))
    y = torch.rand((32, 168, 4))
    model = Autoformer(seq_len=96, pred_len=168, sliding_size=25, d_model=128, n_head=4, c_out=4)
    pred, attn = model(x, y)
    print(pred.shape, attn.shape)
