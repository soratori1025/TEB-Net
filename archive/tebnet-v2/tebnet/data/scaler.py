import torch
class StandardScaler:
    def __init__(self, mean_cp, std_cp, mean_h, std_h, mean_s, std_s):
        self.mean_cp = mean_cp
        self.std_cp = std_cp
        self.mean_h = mean_h
        self.std_h = std_h
        self.mean_s = mean_s
        self.std_s = std_s
    def transform(self, cp, h, s):
        cp_scaled = (cp - self.mean_cp) / self.std_cp
        h_scaled = (h - self.mean_h) / self.std_h
        s_scaled = (s - self.mean_s) / self.std_s
        return cp_scaled, h_scaled, s_scaled
    def inverse_transform(self, cp_scaled, h_scaled, s_scaled):
        cp = cp_scaled * self.std_cp + self.mean_cp
        h = h_scaled * self.std_h + self.mean_h
        s = s_scaled * self.std_s + self.mean_s
        return cp, h, s
    def to(self, device):
        self.mean_cp = self.mean_cp.to(device)
        self.std_cp = self.std_cp.to(device)
        self.mean_h = self.mean_h.to(device)
        self.std_h = self.std_h.to(device)
        self.mean_s = self.mean_s.to(device)
        self.std_s = self.std_s.to(device)
        return self
