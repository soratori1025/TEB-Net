from __future__ import annotations
import math
import torch
from torch import Tensor, nn
R_GAS = 8.314462618
class NASA7ThermodynamicLayer(nn.Module):
    def __init__(self, t_ref: float = 298.15, t_mid: float = 1000.0):
        super().__init__()
        self.t_ref = t_ref
        self.t_mid = t_mid
    def forward(self, a1_5_low: Tensor, a1_5_high: Tensor, H298: Tensor, S298: Tensor, T_grid: Tensor) -> dict[str, Tensor]:
        orig_dtype = a1_5_low.dtype
        a1_5_low  = a1_5_low.double()
        a1_5_high = a1_5_high.double()
        H298      = H298.double()
        S298      = S298.double()
        T_grid    = T_grid.double()
        a1_l, a2_l, a3_l, a4_l, a5_l = [a1_5_low[:, i] for i in range(5)]
        poly_h_ref = a1_l + (a2_l * self.t_ref) / 2 + (a3_l * self.t_ref**2) / 3 + \
                     (a4_l * self.t_ref**3) / 4 + (a5_l * self.t_ref**4) / 5
        a6_l = self.t_ref * (H298 / (R_GAS * self.t_ref) - poly_h_ref)
        poly_s_ref = a1_l * torch.log(torch.tensor(self.t_ref, device=H298.device)) + \
                     a2_l * self.t_ref + (a3_l * self.t_ref**2) / 2 + \
                     (a4_l * self.t_ref**3) / 3 + (a5_l * self.t_ref**4) / 4
        a7_l = S298 / R_GAS - poly_s_ref
        poly_h_mid_l = a1_l + (a2_l * self.t_mid) / 2 + (a3_l * self.t_mid**2) / 3 + \
                       (a4_l * self.t_mid**3) / 4 + (a5_l * self.t_mid**4) / 5
        H_mid_over_RTm = poly_h_mid_l + a6_l / self.t_mid
        poly_s_mid_l = a1_l * math.log(self.t_mid) + \
                       a2_l * self.t_mid + (a3_l * self.t_mid**2) / 2 + \
                       (a4_l * self.t_mid**3) / 3 + (a5_l * self.t_mid**4) / 4
        S_mid_over_R = poly_s_mid_l + a7_l
        a2_h, a3_h, a4_h, a5_h = [a1_5_high[:, i] for i in range(1, 5)]
        cp_low_mid_over_r = (a1_l + a2_l*self.t_mid + a3_l*self.t_mid**2
                             + a4_l*self.t_mid**3 + a5_l*self.t_mid**4)
        a1_h = cp_low_mid_over_r - (a2_h*self.t_mid + a3_h*self.t_mid**2
                                    + a4_h*self.t_mid**3 + a5_h*self.t_mid**4)
        poly_h_mid_h = a1_h + (a2_h * self.t_mid) / 2 + (a3_h * self.t_mid**2) / 3 + \
                       (a4_h * self.t_mid**3) / 4 + (a5_h * self.t_mid**4) / 5
        a6_h = self.t_mid * (H_mid_over_RTm - poly_h_mid_h)
        poly_s_mid_h = a1_h * math.log(self.t_mid) + \
                       a2_h * self.t_mid + (a3_h * self.t_mid**2) / 2 + \
                       (a4_h * self.t_mid**3) / 3 + (a5_h * self.t_mid**4) / 4
        a7_h = S_mid_over_R - poly_s_mid_h
        batch_size = a1_5_low.shape[0]
        n_temps = len(T_grid)
        Cp_grid = torch.zeros((batch_size, n_temps), device=H298.device, dtype=torch.float64)
        H_grid  = torch.zeros((batch_size, n_temps), device=H298.device, dtype=torch.float64)
        S_grid  = torch.zeros((batch_size, n_temps), device=H298.device, dtype=torch.float64)
        low_mask = T_grid <= self.t_mid
        high_mask = T_grid > self.t_mid
        def compute_properties(mask, a1, a2, a3, a4, a5, a6, a7):
            if not mask.any():
                return
            T = T_grid[mask]
            T_uns = T.unsqueeze(0)
            a1_e, a2_e, a3_e, a4_e, a5_e, a6_e, a7_e = [
                a.unsqueeze(-1) for a in (a1, a2, a3, a4, a5, a6, a7)
            ]
            Cp = R_GAS * (a1_e + a2_e*T_uns + a3_e*T_uns**2 + a4_e*T_uns**3 + a5_e*T_uns**4)
            H = R_GAS * T_uns * (a1_e + a2_e*T_uns/2 + a3_e*T_uns**2/3 + a4_e*T_uns**3/4 + a5_e*T_uns**4/5 + a6_e/T_uns)
            S = R_GAS * (a1_e*torch.log(T_uns) + a2_e*T_uns + a3_e*T_uns**2/2 + a4_e*T_uns**3/3 + a5_e*T_uns**4/4 + a7_e)
            Cp_grid[:, mask] = Cp
            H_grid[:, mask] = H
            S_grid[:, mask] = S
        compute_properties(low_mask, a1_l, a2_l, a3_l, a4_l, a5_l, a6_l, a7_l)
        compute_properties(high_mask, a1_h, a2_h, a3_h, a4_h, a5_h, a6_h, a7_h)
        coeffs_low = torch.stack([a1_l, a2_l, a3_l, a4_l, a5_l, a6_l, a7_l], dim=-1)
        coeffs_high = torch.stack([a1_h, a2_h, a3_h, a4_h, a5_h, a6_h, a7_h], dim=-1)
        return {
            "cp": Cp_grid.to(orig_dtype),
            "H":  H_grid.to(orig_dtype),
            "S":  S_grid.to(orig_dtype),
            "coeffs_low":  coeffs_low.to(orig_dtype),
            "coeffs_high": coeffs_high.to(orig_dtype),
            "dh_f_298": H298.to(orig_dtype),
            "s_298": S298.to(orig_dtype),
        }
