"""Climate Net - Pytorch Model"""

import torch
import torch.nn as nn
import numpy as np


class ClimateNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        hidden_layer = 15
        self.inland = nn.Sequential(
            nn.Linear(6, hidden_layer), nn.LeakyReLU(), nn.LayerNorm(hidden_layer),
            nn.Linear(hidden_layer, hidden_layer), nn.LeakyReLU(), nn.LayerNorm(hidden_layer),
            nn.Linear(hidden_layer, hidden_layer), nn.LeakyReLU(), nn.LayerNorm(hidden_layer),
            nn.Linear(hidden_layer, hidden_layer), nn.LeakyReLU(), nn.LayerNorm(hidden_layer),
            nn.Linear(hidden_layer, 2), nn.Tanhshrink()
        ).to(device)

        hidden_layer = 15
        self.coastline_left = nn.Sequential(
            nn.Linear(6, hidden_layer), nn.LeakyReLU(), nn.LayerNorm(hidden_layer),
            nn.Linear(hidden_layer, hidden_layer), nn.LeakyReLU(), nn.LayerNorm(hidden_layer),
            nn.Linear(hidden_layer, hidden_layer), nn.LeakyReLU(), nn.LayerNorm(hidden_layer),
            nn.Linear(hidden_layer, 2), nn.Tanhshrink()
        ).to(device)
        self.coastline_right = nn.Sequential(
            nn.Linear(6, hidden_layer), nn.LeakyReLU(), nn.LayerNorm(hidden_layer),
            nn.Linear(hidden_layer, hidden_layer), nn.LeakyReLU(), nn.LayerNorm(hidden_layer),
            nn.Linear(hidden_layer, hidden_layer), nn.LeakyReLU(), nn.LayerNorm(hidden_layer),
            nn.Linear(hidden_layer, 2), nn.Tanhshrink()
        ).to(device)
        self.coastline_compare_left = nn.Sequential(
            nn.Linear(3, hidden_layer), nn.LeakyReLU(), nn.LayerNorm(hidden_layer),
            nn.Linear(hidden_layer, hidden_layer), nn.LeakyReLU(), nn.LayerNorm(hidden_layer),
            nn.Linear(hidden_layer, hidden_layer), nn.LeakyReLU(), nn.LayerNorm(hidden_layer),
            nn.Linear(hidden_layer, 1), nn.Tanhshrink()
        ).to(device)
        self.coastline_compare_right = nn.Sequential(
            nn.Linear(3, hidden_layer), nn.LeakyReLU(), nn.LayerNorm(hidden_layer),
            nn.Linear(hidden_layer, hidden_layer), nn.LeakyReLU(), nn.LayerNorm(hidden_layer),
            nn.Linear(hidden_layer, hidden_layer), nn.LeakyReLU(), nn.LayerNorm(hidden_layer),
            nn.Linear(hidden_layer, 1), nn.Tanhshrink()
        ).to(device)

        hidden_layer = 50
        self.elevation_diffs_left = nn.Sequential(
            nn.Linear(24, hidden_layer), nn.LeakyReLU(), nn.LayerNorm(hidden_layer),
            nn.Linear(hidden_layer, hidden_layer), nn.LeakyReLU(), nn.LayerNorm(hidden_layer),
            nn.Linear(hidden_layer, hidden_layer), nn.LeakyReLU(), nn.LayerNorm(hidden_layer),
            nn.Linear(hidden_layer, hidden_layer), nn.LeakyReLU(), nn.LayerNorm(hidden_layer),
            nn.Linear(hidden_layer, hidden_layer), nn.LeakyReLU(), nn.LayerNorm(hidden_layer),
            nn.Linear(hidden_layer, hidden_layer), nn.LeakyReLU(), nn.LayerNorm(hidden_layer),
            nn.Linear(hidden_layer, 6), nn.Tanhshrink()
        ).to(device)
        self.elevation_diffs_right = nn.Sequential(
            nn.Linear(24, hidden_layer), nn.LeakyReLU(), nn.LayerNorm(hidden_layer),
            nn.Linear(hidden_layer, hidden_layer), nn.LeakyReLU(), nn.LayerNorm(hidden_layer),
            nn.Linear(hidden_layer, hidden_layer), nn.LeakyReLU(), nn.LayerNorm(hidden_layer),
            nn.Linear(hidden_layer, hidden_layer), nn.LeakyReLU(), nn.LayerNorm(hidden_layer),
            nn.Linear(hidden_layer, hidden_layer), nn.LeakyReLU(), nn.LayerNorm(hidden_layer),
            nn.Linear(hidden_layer, hidden_layer), nn.LeakyReLU(), nn.LayerNorm(hidden_layer),
            nn.Linear(hidden_layer, 6), nn.Tanhshrink()
        ).to(device)

        # 8 latitude cells, 6 coastline
        hidden_layer = 24
        self.coastline_latitude = nn.Sequential(
            nn.Linear(14, hidden_layer), nn.LeakyReLU(), nn.LayerNorm(hidden_layer),
            nn.Linear(hidden_layer, hidden_layer), nn.LeakyReLU(), nn.LayerNorm(hidden_layer),
            nn.Linear(hidden_layer, hidden_layer), nn.LeakyReLU(), nn.LayerNorm(hidden_layer),
            nn.Linear(hidden_layer, hidden_layer), nn.LeakyReLU(), nn.LayerNorm(hidden_layer),
            nn.Linear(hidden_layer, hidden_layer), nn.LeakyReLU(), nn.LayerNorm(hidden_layer),
            nn.Linear(hidden_layer, 5), nn.Tanhshrink()
        ).to(device)

        hidden_layer = 40
        self.final = nn.Sequential(
            nn.Linear(2 + 12 + 5 + 3 + 2, hidden_layer), nn.LeakyReLU(), nn.LayerNorm(hidden_layer),
            nn.Linear(hidden_layer, hidden_layer), nn.LeakyReLU(), nn.LayerNorm(hidden_layer),
            nn.Linear(hidden_layer, hidden_layer), nn.LeakyReLU(), nn.LayerNorm(hidden_layer),
            nn.Linear(hidden_layer, hidden_layer), nn.LeakyReLU(), nn.LayerNorm(hidden_layer),
            nn.Linear(hidden_layer, hidden_layer), nn.LeakyReLU(), nn.LayerNorm(hidden_layer),
            nn.Linear(hidden_layer, 24), nn.Tanhshrink()
        ).to(device)

        hidden = 40
        self.range = nn.Sequential(
            nn.Linear(2 + 12 + 5 + 3, hidden_layer), nn.LeakyReLU(), nn.LayerNorm(hidden_layer),
            nn.Linear(hidden_layer, hidden_layer), nn.LeakyReLU(), nn.LayerNorm(hidden_layer),
            nn.Linear(hidden_layer, hidden_layer), nn.LeakyReLU(), nn.LayerNorm(hidden_layer),
            nn.Linear(hidden_layer, hidden_layer), nn.LeakyReLU(), nn.LayerNorm(hidden_layer),
            nn.Linear(hidden_layer, hidden_layer), nn.LeakyReLU(), nn.LayerNorm(hidden_layer),
            nn.Linear(hidden_layer, 2), nn.Tanhshrink()
        ).to(device)


    def forward(self, x: torch.Tensor, train: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        x = x.float().to(device)
        inland = self.inland(x[:, 11:17])
        coastline_left = self.coastline_left(x[:, np.array([17, 19, 21, 23, 25, 27])])
        coastline_right = self.coastline_right(x[:, np.array([18, 20, 22, 24, 26, 28])])
        coastline_compare_left = self.coastline_compare_left(x[:, np.array([29, 31, 33])])
        coastline_compare_right = self.coastline_compare_right(x[:, np.array([30, 32, 34])])
        elevation_diffs_left = self.elevation_diffs_left(x[:, np.array(
            [37, 38, 42, 43, 44, 48, 59, 50, 54, 55, 56, 60, 61, 62, 66, 67, 68, 72, 73, 74, 78, 79,
             80, 84])])
        elevation_diffs_right = self.elevation_diffs_left(x[:, np.array(
            [40, 39, 41, 46, 45, 47, 52, 51, 53, 58, 57, 59, 64, 63, 65, 70, 69, 71, 76, 75, 77, 82,
             81, 83])])

        combined = torch.concat(
            [x[:, 3:11], coastline_left, coastline_right, coastline_compare_left,
             coastline_compare_right], dim=1)
        coastline_latitude = self.coastline_latitude(combined)
        semifinal = torch.concat(
            [coastline_latitude, elevation_diffs_left, elevation_diffs_right, inland, x[:, :3]],
            dim=1)
        ranges = self.range(semifinal)
        final = self.final(torch.concat([semifinal, ranges], dim=1))

        final[:, 12:24] = torch.exp(final[:, 12:24])

        if train:
            return final, ranges
        return final


device = 'cuda' if torch.cuda.is_available() else 'cpu'
climate_net = ClimateNet()
climate_net.load_state_dict(torch.load('climate_net_params.pt', map_location=device))
climate_net.eval()
