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
    
def hyperfanin_init_weight(linear_layer, hypernet_in, mainnet_in):
    bound = 1e-3 * math.sqrt(3 / (hypernet_in * mainnet_in))
    nn.init.uniform_(linear_layer.weight, -bound, bound)
    nn.init.constant_(linear_layer.bias, 0.0)


def hyperfanin_init_bias(linear_layer, hypernet_in):
    bound = 1e-3 * math.sqrt(3 / (hypernet_in))
    nn.init.uniform_(linear_layer.weight, -bound, bound)
    nn.init.constant_(linear_layer.bias, 0.0)


class SimpleGenerator(nn.Module):
    def __init__(self, compress_dim, hidden_size, lora_targets, lora_rank):
        super(SimpleGenerator, self).__init__()

        self.compress_dim = compress_dim
        self.hidden_size = hidden_size
        self.lora_rank = lora_rank

        hidden_dim = 4 * hidden_size
        multiple_of = 1024
        ffn_dim_multiplier = 1.3
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        
        self.activation_fn = nn.ReLU()
        self.dropout = nn.Dropout(p=0.05)

        self.lora_targets = lora_targets
        # output weights
        if 'Q' in lora_targets:
            self.q_l1 = nn.Linear(self.compress_dim, self.hidden_size * self.lora_rank)
            hyperfanin_init_weight(self.q_l1, self.compress_dim, self.hidden_size * self.lora_rank)
        if 'K' in lora_targets:
            self.k_l1 = nn.Linear(self.compress_dim, self.hidden_size * self.lora_rank)
            hyperfanin_init_weight(self.k_l1, self.compress_dim, self.hidden_size * self.lora_rank)
        if 'V' in lora_targets:
            self.v_l1 = nn.Linear(self.compress_dim, self.hidden_size * self.lora_rank)
            hyperfanin_init_weight(self.v_l1, self.compress_dim, self.hidden_size * self.lora_rank)
        if 'O' in lora_targets:
            self.o_l1 = nn.Linear(self.compress_dim, self.hidden_size * self.lora_rank)
            hyperfanin_init_weight(self.o_l1, self.compress_dim, self.hidden_size * self.lora_rank)
        if 'FFN_UP' in lora_targets:
            self.ffn_up_l1 = nn.Linear(self.compress_dim, self.hidden_size * self.lora_rank)
            hyperfanin_init_weight(self.ffn_up_l1, self.compress_dim, self.hidden_size * self.lora_rank)
        if 'FFN_DOWN' in lora_targets:
            self.ffn_down_l1 = nn.Linear(self.compress_dim, hidden_dim * self.lora_rank)
            hyperfanin_init_weight(self.ffn_down_l1, self.compress_dim, hidden_dim * self.lora_rank)

    def forward(self, x):
        x = self.activation_fn(x)
        x = self.dropout(x)
        q,k,v,o,up,down = None,None,None,None,None,None
        if 'Q' in self.lora_targets:
            q = self.q_l1(x)
        if 'K' in self.lora_targets:
            k = self.k_l1(x)
        if 'V' in self.lora_targets:
            v = self.v_l1(x)
        if 'O' in self.lora_targets:
            o = self.o_l1(x)
        if 'FFN_UP' in self.lora_targets:
            up = self.ffn_up_l1(x)
        if 'FFN_DOWN' in self.lora_targets:
            down = self.ffn_down_l1(x)
        return (q,k,v,o,up,down)


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


class LoraParameterGenerator(nn.Module):
    def __init__(self, layers_num, embed_size, compress_dim, hidden_size, lora_targets=None, lora_rank=None, common_encoder=False, serial_generate=False, encoder_layer_num=1):
        super(LoraParameterGenerator, self).__init__()
        self.layers_num = layers_num
        self.embed_size = embed_size
        self.compress_dim = compress_dim
        self.hidden_size = hidden_size
        self.lora_rank = lora_rank
        lora_targets = lora_targets.split(',')
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

        self.decoder = SimpleGenerator(self.compress_dim, self.hidden_size, lora_targets, self.lora_rank)
        
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