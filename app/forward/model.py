import torch
import torch.nn as nn
from collections import OrderedDict

X_MIN = -5
X_MAX = 5

Y_MIN = -5
Y_MAX = 5

T_MIN = 0
T_MAX = 1

OX_MEAN = 3
OX_STD = 2

OY_MEAN = 3
OY_STD = 2

G_MEAN = 2
G_STD = 1

PSI_MEAN = 0.001
PSI_STD = 0.1

class GrossPitaevskiiDNN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.depth = len(layers) - 1
        self.activation = torch.nn.Tanh

        layer_list = []
        for i in range(self.depth - 1):
            layer_list.append((f"layer_{i}", nn.Linear(layers[i], layers[i+1])))
            layer_list.append((f"activation_{i}", self.activation()))

        layer_list.append((f"layer_{self.depth - 1}", nn.Linear(layers[-2], layers[-1])))

        self.layers = nn.Sequential(OrderedDict(layer_list))

    def forward(self, x):
        out = self.layers[0](x)
        out_in = 0
        for i in range(1, len(self.layers)):
            if (i % 4) == 1:
                out = self.layers[i](out)
                out_in = out
            elif (i % 4) == 0:
                out = out_in + self.layers[i](out)
            else:
                out = self.layers[i](out)
        return out


class GrossPitaevskiiInference:
    def __init__(self, checkpoint_path):
        self.x_min, self.x_max = X_MIN, X_MAX
        self.y_min, self.y_max = Y_MIN, Y_MAX
        self.t_min, self.t_max = T_MIN, T_MAX

        self.ox_mean, self.ox_std = OX_MEAN, OX_STD
        self.oy_mean, self.oy_std = OY_MEAN, OY_STD
        self.g_mean, self.g_std = G_MEAN, G_STD
        self.psi_mean, self.psi_std = PSI_MEAN, PSI_STD

        self.model = GrossPitaevskiiDNN(
            layers=[6, 250, 250, 250, 250, 250, 250, 250, 2]
        )

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()


    def normalize(self, v, vmin, vmax):
        return (v - vmin) / (vmax - vmin) * 2 - 1

    def wave_function(self, x, y, t, omega_x, omega_y, g):
        inputs = torch.cat([x, y, t, omega_x, omega_y, g], dim=1)
        out = self.model(inputs)
        return out[:, 0:1], out[:, 1:2]

    def predict(self, x, y, t, omega_x, omega_y, g):
        with torch.no_grad():
            x_n = self.normalize(x, self.x_min, self.x_max)
            y_n = self.normalize(y, self.y_min, self.y_max)
            t_n = self.normalize(t, self.t_min, self.t_max)

            ox_n = (omega_x - self.ox_mean) / self.ox_std
            oy_n = (omega_y - self.oy_mean) / self.oy_std
            g_n  = (g - self.g_mean) / self.g_std

            psi_r, psi_i = self.wave_function(x_n, y_n, t_n, ox_n, oy_n, g_n)

            psi_r = psi_r * self.psi_std + self.psi_mean
            psi_i = psi_i * self.psi_std + self.psi_mean

            return torch.cat([psi_r, psi_i], dim=1)