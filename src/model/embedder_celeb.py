import torch
import torch.nn as nn
from .utils import positional_encoding_init, build_mlp, build_mlp_with_linear_skipcon


class NonLinearEmbedder(
    nn.Module
):  # This will no longer work! Need to fix the inputs to mimic the other classes
    """
    Non-Linear TNPD embedder used in TNP paper
    """

    def __init__(self, dim_xc, dim_yc, num_latent, dim_hid, dim_out, emb_depth, name):
        super().__init__()

        # -1 for removing markers
        self.embedder = build_mlp(dim_xc - 1 + dim_yc, dim_hid, dim_out, emb_depth)
        self.name = name

    def construct_input(self, batch):
        """
        Construct the input before feeding it to the embedder
        """
        # remove markers for fair comparison with ours
        x_y_ctx = torch.cat((batch.xc[:, :, 1:], batch.yc), dim=-1)
        x_0_tar = torch.cat((batch.xt[:, :, 1:], torch.zeros_like(batch.yt)), dim=-1)
        inp = torch.cat((x_y_ctx, x_0_tar), dim=1)
        return inp

    def forward(self, batch):
        """
        Embedder call
        """
        inp = self.construct_input(batch)
        return self.embedder(inp)


class EmbedderMarker(nn.Module):
    def __init__(
        self,
        dim_xc,
        dim_yc,
        num_latent,
        dim_hid,
        dim_out,
        emb_depth,
        name=None,
        pos_emb_init=False,
        use_skipcon_mlp=False,
        discrete_index = None,
    ):
        super().__init__()
        self.embedder_marker = nn.Embedding(2 + num_latent, dim_out)  # E_c
        if discrete_index:
            self.embedder_discrete = nn.Embedding(len(discrete_index), dim_out)

        if use_skipcon_mlp:
            self.embedderx = build_mlp_with_linear_skipcon(
                dim_xc - 1, dim_hid, dim_out, emb_depth
            )  # f_cov
            self.embedderyc = build_mlp_with_linear_skipcon(
                dim_yc, dim_hid, dim_out, emb_depth
            )  # f_val
        else:
            self.embedderx = build_mlp(dim_xc - 1, dim_hid, dim_out, emb_depth)  # f_cov
            self.embedderyc = build_mlp(dim_yc, dim_hid, dim_out, emb_depth)  # f_val

        self.name = name
        self.discrete_index = discrete_index

        # positional embedding initialization
        if pos_emb_init:
            self.embedder_marker.weight = positional_encoding_init(
                2 + num_latent, dim_out, 2 + num_latent
            )

    def forward(self, batch):
        # xc and yc is context:
        # xc is either a continuous variable or 0 for latent
        # xce is a marker 1 for data >1 for latent

        if torch.any(torch.isnan(batch.yc)):
            print("yc contains NaN values:", batch.yc)

        # batch.xc [B, T, [marker, x1, x2, x3, ...]] where marker is 1 for data.
        # batch.xc [B, T, [marker, 0, 0, ...]] for latent marker  > 1 for latent.
        xc = batch.xc[:, :, 1:] 
        xce = batch.xc[:, :, :1]
        yc = batch.yc

        xt = batch.xt[:, :, 1:]
        xte = batch.xt[:, :, :1]
        yt = batch.yt

        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create the mask
        if self.discrete_index:
            discrete_values = torch.tensor(self.discrete_index, device=device)
            mask_discrete = torch.isin(batch.yc, torch.tensor(discrete_values)).float()[:,:,:1]
            inverse_mask_discrete = 1 - mask_discrete
        else:
            mask_discrete = torch.zeros_like(batch.yc[:,:,:1], device=device).float()
            

        mask_context_x = (xce.int() == 1).float()
        
        if self.discrete_index:
            context_embedding = (
                torch.add(
                    self.embedderx(xc) * mask_context_x,  # Embedd for continuous x data
                    self.embedderyc(yc) * inverse_mask_discrete,  # Embedd continuous y data
                )
                + self.embedder_discrete((yc[:,:,-1:] * mask_discrete)[:, :, 0].int()) #discrete just last index
                * mask_discrete
            )  # Embedd discrete y data
        else:
            context_embedding = torch.add(
                self.embedderx(xc) * mask_context_x,
                self.embedderyc(yc),
            )

        context_embedding = torch.add(
            context_embedding, self.embedder_marker(xce[:, :, 0].int())
        )  # Embedd for markers [1,2,3,...]

        # add xt and marker_? embeddings
        # set embedderx output to zero when point in target set is latent
        mask_target_x = (xte.int() == 1).float()

        target_embedding = torch.add(
            self.embedderx(xt) * mask_target_x,
            self.embedder_marker(
                torch.zeros_like(yt).int()[
                    :, :, 0
                ]  # we use the 0th index of the embedder for "?" marker.
            ),
        )

        target_embedding = torch.add(
            target_embedding, self.embedder_marker(xte[:, :, 0].int()) # Embedd for markers [1,2,3,...]
        )  # Embedd for markers [1,2,3,...]

        res = torch.cat((context_embedding, target_embedding), dim=1)
        return res
