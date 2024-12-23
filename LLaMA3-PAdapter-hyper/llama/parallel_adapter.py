import torch.nn as nn
import torch
import math

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
    
class LlamaAdapterLayer(nn.Module):
    def __init__(self, hidden_size, adapter_size, hyper_padapter=False):
        super(LlamaAdapterLayer, self).__init__()
        self.hidden_size = hidden_size
        self.adapter_size = adapter_size
        self.adapter_down_proj_weight = None
        self.adapter_down_proj_bias = None
        self.adapter_up_proj_weight = None
        self.adapter_up_proj_bias = None
        self.adapter_act_fn = nn.SiLU()

        self.hyper_padapter = hyper_padapter
        # not use
        self.not_use = nn.Linear(1, 1)
        if not self.hyper_padapter:
            self.adapter_down_manual_proj = nn.Linear(hidden_size, adapter_size)
            self.adapter_up_manual_proj = nn.Linear(adapter_size, hidden_size)
            nn.init.xavier_uniform_(self.adapter_down_manual_proj.weight, gain=1e-4)
            nn.init.xavier_uniform_(self.adapter_up_manual_proj.weight, gain=1e-4)
            nn.init.constant_(self.adapter_down_manual_proj.bias, 0.0)
            nn.init.constant_(self.adapter_up_manual_proj.bias, 0.0)
            
    def clear_adapter(self):
        self.adapter_down_proj_weight = None
        self.adapter_down_proj_bias = None
        self.adapter_up_proj_weight = None
        self.adapter_up_proj_bias = None

    def apply_adapter_params(self, dw, db, uw, ub):
        batch_size = dw.shape[0]
        self.adapter_down_proj_weight = dw.view(batch_size, self.hidden_size, self.adapter_size)
        self.adapter_down_proj_bias = db.view(batch_size, self.adapter_size)
        self.adapter_up_proj_weight = uw.view(batch_size, self.adapter_size, self.hidden_size)
        self.adapter_up_proj_bias = ub.view(batch_size, self.hidden_size)
        
    def forward(self, x):
        if self.adapter_down_proj_weight is not None:
            x = (x @ self.adapter_down_proj_weight) + self.adapter_down_proj_bias.unsqueeze(1)
            x = self.adapter_act_fn(x)
            x = (x @ self.adapter_up_proj_weight) + self.adapter_up_proj_bias.unsqueeze(1)
        else:
            x = self.adapter_down_manual_proj(x)
            x = self.adapter_act_fn(x)
            x = self.adapter_up_manual_proj(x)
        return x


def hyperfanin_init_weight(linear_layer, hypernet_in, mainnet_in):
    bound = 1e-3 * math.sqrt(3 / (hypernet_in * mainnet_in))
    nn.init.uniform_(linear_layer.weight, -bound, bound)
    nn.init.constant_(linear_layer.bias, 0.0)


def hyperfanin_init_bias(linear_layer, hypernet_in):
    bound = 1e-3 * math.sqrt(3 / (hypernet_in))
    nn.init.uniform_(linear_layer.weight, -bound, bound)
    nn.init.constant_(linear_layer.bias, 0.0)


class SimpleGenerator(nn.Module):
    def __init__(self, compress_dim, hidden_size, adapter_size):
        super(SimpleGenerator, self).__init__()

        self.compress_dim = compress_dim
        self.hidden_size = hidden_size
        self.adapter_size = adapter_size
        
        self.activation_fn = nn.ReLU()
        # output weights
        self.weight_down = nn.Linear(self.compress_dim, self.hidden_size * self.adapter_size)
        self.weight_up = nn.Linear(self.compress_dim, self.hidden_size * self.adapter_size)
        self.bias_down = nn.Linear(self.compress_dim, self.adapter_size)
        self.bias_up = nn.Linear(self.compress_dim, self.hidden_size)
        # init weights
        hyperfanin_init_weight(self.weight_down, self.compress_dim, self.adapter_size * self.hidden_size)
        hyperfanin_init_weight(self.weight_up, self.compress_dim, self.adapter_size * self.hidden_size)
        hyperfanin_init_bias(self.bias_down, self.hidden_size)
        hyperfanin_init_bias(self.bias_up, self.hidden_size)

        self.dropout = nn.Dropout(p=0.05)

    def forward(self, x):
        x = self.activation_fn(x)
        x = self.dropout(x)
        return (
            self.weight_down(x),
            self.bias_down(x),
            self.weight_up(x),
            self.bias_up(x),
        )


class Encoder(nn.Module):
    def __init__(self, input_dim, compress_dim, encoder_layer_num=1, dropout=0.05):
        super(Encoder, self).__init__()

        self.input_dim = input_dim
        self.compress_dim = compress_dim
        self.encoder_layer_num = encoder_layer_num
        

        if encoder_layer_num == 1:
            self.linear0 = nn.Linear(input_dim, compress_dim)
            hyperfanin_init_weight(self.linear0, input_dim, compress_dim)
        else:
            middle_dim = compress_dim
            self.linear0 = nn.Linear(input_dim, middle_dim)
            self.activation_fn = nn.ReLU()
            self.dropout = nn.Dropout(p=dropout)
            self.linear1 = nn.Linear(middle_dim, compress_dim)
            hyperfanin_init_weight(self.linear0, input_dim, middle_dim)
            hyperfanin_init_weight(self.linear1, middle_dim, compress_dim)

    def forward(self, x):
        if self.encoder_layer_num == 1:
            x = self.linear0(x)
        else:
            x = self.linear0(x)
            x = self.dropout(self.activation_fn(x))
            x = self.linear1(x)
        return x


class LlamaParameterGenerator(nn.Module):
    def __init__(self, layers_num, embed_size, compress_dim, hidden_size, adapter_size, common_encoder=False, serial_generate=False, encoder_layer_num=1):
        super(LlamaParameterGenerator, self).__init__()
        self.layers_num = layers_num
        self.embed_size = embed_size
        self.compress_dim = compress_dim
        self.hidden_size = hidden_size
        self.adapter_size = adapter_size
        
        self.common_encoder = common_encoder
        self.serial_generate = serial_generate
        input_dim = self.hidden_size + self.embed_size
        self.norm = RMSNorm(input_dim)
        self.dropout = nn.Dropout(p=0.05)
        if serial_generate and not common_encoder: # serial generate and different encoder
            self.encoders = nn.ModuleList()
            for i in range(layers_num):
                encoder = Encoder(input_dim, self.compress_dim, encoder_layer_num=encoder_layer_num)
                self.encoders.append(encoder)
        else:
            if common_encoder:
                self.encoder_num = 1
            else:
                self.encoder_num = self.layers_num
            self.encoder = Encoder(input_dim, self.encoder_num * self.compress_dim, encoder_layer_num=encoder_layer_num)

        self.decoder = SimpleGenerator(self.compress_dim, self.hidden_size, self.adapter_size)
        
    def forward(self, hidden_input, hyper_index=None):
        # [batch, input_dim]
        batch_size = hidden_input.shape[0]
        hidden_input = self.norm(hidden_input)
        hidden_input = self.dropout(hidden_input)
        if self.serial_generate:
            if self.common_encoder:
                compress_hidden = self.encoder(hidden_input)
            else: # different encoder
                compress_hidden = self.encoders[hyper_index](hidden_input)
            return self.decoder(compress_hidden)
        else:
            layers = []
            compress_hiddens = self.encoder(hidden_input) # [batch, compress_dim * encoder_num]
            if self.common_encoder:
                for i in range(self.layers_num):
                    compress_hidden = compress_hiddens
                    layers.append(self.decoder(compress_hidden))
            else: # different encoder
                compress_hiddens  = compress_hiddens.view(batch_size, self.layers_num, self.compress_dim) # [batch, encoder_num, compress_dim]
                for i in range(self.layers_num):
                    compress_hidden = compress_hiddens[:, i]
                    layers.append(self.decoder(compress_hidden))
            return layers