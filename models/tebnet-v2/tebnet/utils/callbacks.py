from __future__ import annotations
import logging
import lightning.pytorch as pl
import torch

logger = logging.getLogger(__name__)

R_GAS = 8.314462618

def _h_over_rt(T, c):
    a1,a2,a3,a4,a5,a6 = [c[:, i:i+1] for i in range(6)]
    return a1 + a2*T/2 + a3*T**2/3 + a4*T**3/4 + a5*T**4/5 + a6/T

def _s_over_r(T, c):
    a1,a2,a3,a4,a5 = [c[:, i:i+1] for i in range(5)]
    return a1*torch.log(T) + a2*T + a3*T**2/2 + a4*T**3/3 + a5*T**4/4 + c[:, 6:7]

class ThermodynamicConsistencyCallback(pl.Callback):
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if batch_idx > 0:
            return
        with torch.no_grad():
            preds = pl_module(batch)
        cl, ch = preds["coeffs_low"], preds["coeffs_high"]
        H298, S298 = preds["dh_f_298"], preds["s_298"]
        dev = cl.device
        tref = torch.tensor(pl_module.thermo_layer.t_ref, device=dev)
        tmid = torch.tensor(pl_module.thermo_layer.t_mid, device=dev)
        # (1) Neo: H,S dựng lại tại T_ref phải khớp H298,S298
        anchor_h = (R_GAS*tref*_h_over_rt(tref, cl).squeeze(-1) - H298).abs().mean()
        anchor_s = (R_GAS*_s_over_r(tref, cl).squeeze(-1) - S298).abs().mean()
        # (2) Liên tục tại T_mid: nhánh low vs high
        cont_h = (R_GAS*tmid*(_h_over_rt(tmid, cl) - _h_over_rt(tmid, ch)).squeeze(-1)).abs().mean()
        cont_s = (R_GAS*(_s_over_r(tmid, cl) - _s_over_r(tmid, ch)).squeeze(-1)).abs().mean()
        bs = len(batch)
        pl_module.log("val/anchor_h298", anchor_h, batch_size=bs)
        pl_module.log("val/anchor_s298", anchor_s, batch_size=bs)
        pl_module.log("val/continuity_h_tmid", cont_h, batch_size=bs)
        pl_module.log("val/continuity_s_tmid", cont_s, batch_size=bs)

class PrintValidationMetricsCallback(pl.Callback):
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch
        if "val/mae_cp" in metrics:
            print("\n" + "-" * 80)
            print(f" VALIDATION RESULTS - EPOCH {epoch}")
            print(f"  Cp MAE   : {metrics.get('val/mae_cp', 0.0):.4f} J/(mol K)  |  Huber : {metrics.get('val/huber_cp', 0.0):.4f}")
            print(f"  H298 MAE : {metrics.get('val/mae_h298', 0.0):.1f} J/mol  |  Huber : {metrics.get('val/huber_h298', 0.0):.1f}")
            print(f"  S298 MAE : {metrics.get('val/mae_s298', 0.0):.4f} J/(mol K)  |  Huber : {metrics.get('val/huber_s298', 0.0):.4f}")
            print(f"  Total Loss: {metrics.get('val/loss_total', 0.0):.5f}")
            print("-" * 80 + "\n")
