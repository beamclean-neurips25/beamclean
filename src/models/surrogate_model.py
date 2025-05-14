from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from pathlib import Path


# Base class for Noise Models
class NoiseModelBase(nn.Module):
    def __init__(self, mean: float = 0.0, variance: float = 1.0):
        super(NoiseModelBase, self).__init__()
        self.mean = nn.Parameter(torch.tensor(mean))
        logvar = torch.log(torch.tensor(variance))
        self.log_variance = nn.Parameter(logvar)

    def forward(self, x: torch.Tensor):
        mean = self.mean.expand_as(x)
        logvar = self.log_variance.expand_as(x)
        return mean, logvar

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))


# Create an abstract Mimic Model Class using ABC Abstract Base Class
class SurrogateModelBase(ABC, nn.Module):
    def __init__(self):
        super(SurrogateModelBase, self).__init__()

    def save_model(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        return super().load_state_dict(state_dict, strict, assign)

    @abstractmethod
    def on_epoch_end(self):
        raise NotImplementedError

    @abstractmethod
    def on_train_end(self, results_folder_dir: Path):
        raise NotImplementedError


class SurrogateModelConstant(SurrogateModelBase):
    def __init__(
        self,
        mean: float = 0.0,
        std: float = 1.0,
    ):
        super().__init__()
        self.mean = nn.Parameter(torch.tensor(mean))
        self.log_std = nn.Parameter(torch.log(torch.tensor(std)))
        self.estimated_mean_list = [mean]
        self.estimated_std_list = [std]

    def forward(self, x: torch.Tensor, **_kwargs):
        mean = self.mean.expand_as(x)
        log_std = self.log_std.expand_as(x)
        return mean, log_std

    def on_epoch_end(self) -> dict:
        pass
        # mean = self.mean.item()
        # variance = torch.exp(self.log_var).item()
        # self.estimated_mean_list.append(mean)
        # self.estimated_variance_list.append(variance)
        # return {
        #     "log_dict": {
        #         "Estimated Mean": mean,
        #         "Estimated Variance": variance,
        #     },
        #     "em_stats": {
        #         "Estimated Mean": mean,
        #         "Estimated Variance": variance,
        #     },
        # }

    def on_train_end(self, results_folder_dir: Path) -> dict:
        pass

    # def _on_train_end(self, results_folder_dir: Path) -> dict:
    #     np.save(
    #         results_folder_dir / FileNames.ESTIMATED_MEANS.value,
    #         np.array(self.estimated_mean_list),
    #     )
    #     np.save(
    #         results_folder_dir / FileNames.ESTIMATED_VARIANCES.value,
    #         np.array(self.estimated_variance_list),
    #     )
    #     # Make a convergence plot
    #     sns.set_theme(style="whitegrid")
    #     plt.figure(figsize=(12, 6))
    #     sns.lineplot(
    #         data=self.estimated_mean_list, label="Estimated Mean", marker="o"
    #     )
    #     sns.lineplot(
    #         data=self.estimated_variance_list,
    #         label="Estimated Variance",
    #         marker="o",
    #     )
    #     plt.axhline(
    #         y=self.true_mean,
    #         color="r",
    #         linestyle="--",
    #         label="True Mean",
    #     )
    #     plt.axhline(
    #         y=self.true_variance,
    #         color="g",
    #         linestyle="--",
    #         label="True Variance",
    #     )
    #     plt.title("EM Algorithm Convergence")
    #     plt.legend()
    #     plt.savefig(results_folder_dir / FileNames.CONVERGENCE_PLOT.value)
    #     wandb.log(
    #         {
    #             "Convergence Plot": wandb.Image(
    #                 str(results_folder_dir / FileNames.CONVERGENCE_PLOT.value)
    #             )
    #         }
    #     )


class ForwardModelMLP(SurrogateModelBase):
    def __init__(self, input_dim: int = 2048, hidden_dims: list = (512,)):
        super().__init__()
        for i, h_dim in enumerate(hidden_dims):
            if i == 0:
                self.layers = nn.ModuleList(
                    [
                        nn.Linear(input_dim, h_dim),
                        nn.ReLU(),
                    ]
                )
            else:
                self.layers.extend(
                    [
                        nn.Linear(hidden_dims[i - 1], h_dim),
                        nn.ReLU(),
                    ]
                )
        self.layers.append(nn.Linear(hidden_dims[-1], 2))

    def forward(self, x: torch.Tensor, *_args, **_kwargs):
        # The first (input dim) elements are the mean,
        # the next (input dim) elements are the variance
        _x = x
        for _layer in self.layers:
            _x: torch.Tensor = _layer(_x)
        mean, logvar = _x.chunk(2, dim=-1)
        mean = mean.expand_as(x)
        logvar = logvar.expand_as(x)
        return mean, logvar

    def on_epoch_end(self):
        return {"log_dict": {}, "em_stats": {}}

    def on_train_end(self, results_folder_dir: Path) -> dict:
        pass


def _create_encoder(model_dim: int, num_heads: int, num_layers: int):
    encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads)
    return nn.TransformerEncoder(encoder_layer, num_layers=num_layers)


def _create_output_layer(model_dim: int, embedding_dim):
    return nn.Linear(model_dim, embedding_dim * 2)


class SurrogateModelTransformer(SurrogateModelBase):
    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        num_layers: int,
        causal: bool = True,
    ):
        super().__init__()
        self.encoder = _create_encoder(model_dim, num_heads, num_layers)
        self.output_layer = _create_output_layer(model_dim, model_dim)
        self.causal = causal

    def forward(self, x: torch.Tensor, mask=None):
        x = self.encoder(x, is_causal=self.causal, mask=mask)
        # Forward pass to get the concatenated mean and log variance
        # Shape: (batch_size, seq_len, embedding_dim * 2) noqa: E501
        x = self.output_layer(x)

        # Split outputs into mean and log variance
        embedding_dim = x.size(-1) // 2
        mean = x[..., :embedding_dim]  # Shape: (batch_size, seq_len, embedding_dim)
        logvar = x[..., embedding_dim:]  # Shape: (batch_size, seq_len, embedding_dim)
        return mean, logvar

    def disable_batch_norm_and_dropout(self):
        """Disable batch normalization and dropout layers."""
        for module in self.modules():
            if isinstance(module, nn.BatchNorm1d | nn.BatchNorm2d | nn.BatchNorm3d):
                module.eval()
            if isinstance(module, nn.Dropout | nn.Dropout2d | nn.Dropout3d):
                module.eval()

    def on_epoch_end(self):
        return {"log_dict": {}, "em_stats": {}}

    def on_train_end(self, results_folder_dir: Path) -> dict:
        pass
