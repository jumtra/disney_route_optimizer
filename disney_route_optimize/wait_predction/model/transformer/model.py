import torch.nn as nn
from torch.nn import (
    LayerNorm,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)

from disney_route_optimize.wait_predction.model.transformer.positional_encoding import (
    PositionalEncoding,
    TokenEmbedding,
)


class Transformer(nn.Module):
    def __init__(
        self,
        num_encoder_layers,
        num_decoder_layers,
        d_model,
        d_input,
        d_output,
        dim_feedforward=512,
        dropout=0.1,
        nhead=8,
    ):
        super(Transformer, self).__init__()

        # エンべディングの定義
        self.token_embedding_src = TokenEmbedding(d_input, d_model)
        self.token_embedding_tgt = TokenEmbedding(d_output, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)

        # エンコーダの定義
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        encoder_norm = LayerNorm(d_model)
        self.transformer_encoder = TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers, norm=encoder_norm
        )

        # デコーダの定義
        decoder_layer = TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        decoder_norm = LayerNorm(d_model)
        self.transformer_decoder = TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers, norm=decoder_norm
        )

        # 出力層の定義
        self.output = nn.Linear(d_model, d_output)

    def forward(self, src, tgt, mask_src, mask_tgt):
        # mask_src, mask_tgtはセルフアテンションの際に未来のデータにアテンションを向けないためのマスク

        embedding_src = self.positional_encoding(self.token_embedding_src(src))
        memory = self.transformer_encoder(embedding_src, mask_src)

        embedding_tgt = self.positional_encoding(self.token_embedding_tgt(tgt))
        outs = self.transformer_decoder(embedding_tgt, memory, mask_tgt)

        output = self.output(outs)
        return output

    def encode(self, src, mask_src):
        return self.transformer_encoder(
            self.positional_encoding(self.token_embedding_src(src)), mask_src
        )

    def decode(self, tgt, memory, mask_tgt):
        return self.transformer_decoder(
            self.positional_encoding(self.token_embedding_tgt(tgt)), memory, mask_tgt
        )
