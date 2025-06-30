# gemini-2.5 first pass code, under review
"""
spherical_attention.py

This file will define the SphericalMultiHeadAttention layer.
"""
import torch
import torch.nn as nn
import s2fft

class SphericalMultiHeadAttention(nn.Module):
    def __init__(self, in_channels, out_channels, L, num_heads, sampling="mw"):
        super(SphericalMultiHeadAttention, self).__init__()
        
        if in_channels % num_heads != 0:
            raise ValueError("in_channels must be divisible by num_heads")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.L = L
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.sampling = sampling

        # Learnable weights for Q, K, V for each head
        self.k_q = nn.Parameter(torch.randn(self.num_heads, self.head_dim, self.head_dim, L, 2 * L - 1, dtype=torch.float32))
        self.k_k = nn.Parameter(torch.randn(self.num_heads, self.head_dim, self.head_dim, L, 2 * L - 1, dtype=torch.float32))
        self.k_v = nn.Parameter(torch.randn(self.num_heads, self.head_dim, self.head_dim, L, 2 * L - 1, dtype=torch.float32))
        
        # Output projection
        self.k_o = nn.Parameter(torch.randn(self.out_channels, self.in_channels, L, 2 * L - 1, dtype=torch.float32))

    def forward(self, f_in_lm):
        # f_in_lm shape: (batch, in_channels, L, 2L-1)
        batch_size, _, _, _ = f_in_lm.shape
        
        f_in_lm = f_in_lm.reshape(batch_size, self.num_heads, self.head_dim, self.L, 2 * self.L - 1)

        # Pre-transform weights to spectral domain
        k_q_lm = self._transform_weights(self.k_q, f_in_lm.device)
        k_k_lm = self._transform_weights(self.k_k, f_in_lm.device)
        k_v_lm = self._transform_weights(self.k_v, f_in_lm.device)
        k_o_lm = self._transform_output_weights(self.k_o, f_in_lm.device)

        head_outputs_lm = []
        for h in range(self.num_heads):
            f_in_head_lm = f_in_lm[:, h, ...]

            # Generate Q, K, V in spectral domain
            F_Q_h = torch.einsum('bclm,cdlm->bdlm', f_in_head_lm, k_q_lm[h])
            F_K_h = torch.einsum('bclm,cdlm->bdlm', f_in_head_lm, k_k_lm[h])
            F_V_h = torch.einsum('bclm,cdlm->bdlm', f_in_head_lm, k_v_lm[h])

            # Generate Dynamic Attention Pattern
            F_A_raw_h = F_Q_h * torch.conj(F_K_h)
            
            A_raw_h_spatial = self._inverse_transform(F_A_raw_h)
            
            # Softmax over spatial dimensions
            A_h_spatial = torch.softmax(A_raw_h_spatial.view(batch_size, self.head_dim, -1), dim=-1).view(A_raw_h_spatial.shape)
            
            F_A_h = self._forward_transform(A_h_spatial)

            # Compute Value-Weighted Sum
            F_fout_h = F_V_h * F_A_h
            head_outputs_lm.append(F_fout_h)
        
        # Concatenate head outputs
        f_out_concat_lm = torch.cat(head_outputs_lm, dim=1)
        
        # Output projection
        f_out_final_lm = torch.einsum('bclm,oclm->bolm', f_out_concat_lm, k_o_lm)

        return f_out_final_lm

    def _transform_output_weights(self, weights, device):
        """Helper to transform output weights to spectral domain."""
        out_c, in_c, L, _ = weights.shape
        weights_lm = torch.zeros(out_c, in_c, L, 2 * L - 1, dtype=torch.complex64, device=device)
        for o in range(out_c):
            for i in range(in_c):
                weights_lm[o, i] = torch.from_numpy(s2fft.forward(weights[o, i].detach().cpu().numpy(), L=self.L, sampling=self.sampling)).to(device)
        return weights_lm

    def _transform_weights(self, weights, device):
        """Helper to transform weights to spectral domain."""
        num_heads, out_c, in_c, L, _ = weights.shape
        weights_lm = torch.zeros(num_heads, out_c, in_c, L, 2 * L - 1, dtype=torch.complex64, device=device)
        for h in range(num_heads):
            for o in range(out_c):
                for i in range(in_c):
                    weights_lm[h, o, i] = torch.from_numpy(s2fft.forward(weights[h, o, i].detach().cpu().numpy(), L=self.L, sampling=self.sampling)).to(device)
        return weights_lm

    def _forward_transform(self, x_spatial):
        """Helper for forward spherical transform."""
        batch_size, channels, L, _ = x_spatial.shape
        x_lm = torch.zeros(batch_size, channels, L, 2 * L - 1, dtype=torch.complex64, device=x_spatial.device)
        for b in range(batch_size):
            for c in range(channels):
                x_lm[b, c] = torch.from_numpy(s2fft.forward(x_spatial[b, c].detach().cpu().numpy(), L=self.L, sampling=self.sampling)).to(x_spatial.device)
        return x_lm

    def _inverse_transform(self, x_lm):
        """Helper for inverse spherical transform."""
        batch_size, channels, L, _ = x_lm.shape
        x_spatial = torch.zeros(batch_size, channels, L, 2 * L - 1, dtype=torch.float32, device=x_lm.device)
        for b in range(batch_size):
            for c in range(channels):
                x_spatial[b, c] = torch.from_numpy(s2fft.inverse(x_lm[b, c].detach().cpu().numpy(), L=self.L, sampling=self.sampling)).to(x_lm.device)
        return x_spatial
