import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch import nn, Tensor
from torch.types import _dtype
from typing import Optional, Union, List
from sklearn.manifold import TSNE
from torchvision.transforms import transforms

from src.utils.config_parser import Config
from lib.libdataset.src.utils.tools import Logger, Tools
from lib.libdataset.config.consts import General as _CG


class Model(nn.Module):

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        Logger.instance().info(f"Model instantiated: {self.__class__.__name__}")

    def forward(self, x):
        pass

    @staticmethod
    def get_batch_size(config: Config) -> int:
        """Get the batch size to feed the model

        Args:
            config (Config)

        Returns:
            batch size (int)
        """

        if config.model.context is None:
            Logger.instance().debug(f"batch size is {config.train_test.batch_size}")
            return config.train_test.batch_size
        
        bs = config.model.context.n_way * (config.model.context.k_shot + config.model.context.k_query)
        Logger.instance().debug(f"using meta-learning model to compute batch size: {bs}")
        return bs
    
    def get_out_size(self, pos: Optional[int]) -> Union[torch.Size, int]:
        """Get the output size of a model
        
        This is useful both to know both for extractors or any other Module. The passed fake tensor is shaped according
        to the batch size.

        Args:
            pos (Optional[int]): if specified, returns the exact size in position `pos`; full torch.Size otherwise

        Returns:
            either an integer for the location specified by `pos` or torch.Size
        """
        
        batch_size = Model.get_batch_size(self.config)
        n_channels = len(self.config.dataset.dataset_mean) if self.config.dataset.dataset_mean is not None else 1
        x = torch.randn(batch_size, n_channels, self.config.dataset.image_size, self.config.dataset.image_size)
        
        with torch.no_grad():
            output = self.forward(x.to(_CG.DEVICE))

        if type(output) is tuple or type(output) is list:
            output = output[0]

        # assuming a flat tensor so that shape = (batch_size, feature_vector, Opt[unpooled], Opt[unpooled])
        if pos is not None:
            if pos > len(output.shape):
                raise ValueError(f"required position {pos}, but model output size has {len(output.shape)} values.")
            return output.shape[pos]
        
        return output.shape
    
    @staticmethod
    def mixup(x: torch.Tensor, shuffle: Optional[torch.Tensor], lam: Optional[float]) -> torch.Tensor:
        # https://www.kaggle.com/code/hocop1/manifold-mixup-using-pytorch
        if shuffle is not None and lam is not None:
            x = lam * x + (1 - lam) * x[shuffle]
        return x

    @staticmethod
    def check_weights_path(relative_path: str):
        Logger.instance().debug(f"You are using a custom feature extractor. Must be in {relative_path}")
        try:
            weights_path = Tools.validate_path(relative_path)
        except FileNotFoundError as fnf:
            Logger.instance().critical(f"{fnf}")
            sys.exit(-1)

        return weights_path
    
    @staticmethod
    def one_hot_encoding(y: Tensor, n_classes: int, dtype: _dtype=torch.float):
        one = 1.0 if dtype is torch.float else int(1)
        one_hot_labels = torch.zeros((y.size(0), n_classes), dtype=dtype, device=_CG.DEVICE)
        one_hot_labels[range(len(y)), y] = one
        
        return one_hot_labels

    @staticmethod
    def one_hot_loss(x: Tensor, y: Tensor, _dim: int) -> Tensor:
        log_p_y = torch.nn.functional.log_softmax(x, dim=_dim)
        loss = -(y * log_p_y + 1e-10).sum(dim=_dim).mean()

        acc = x.argmax(1).eq(y.argmax(1)).float().mean()

        return acc, loss
    
    @staticmethod
    def plot_tsne(
        embeddings: torch.Tensor,
        n_classes: int,
        n_samples: int,
        labels: Optional[Tensor],
        epoch: Optional[int]=None
    ):
        if not len(embeddings.shape) == 2:
            Logger.instance().warning(f"Failed t-sne: embeddings should have 2 dim, have {len(embeddings.shape)} instead")

        if not embeddings.device == torch.device("cpu"):
            embeddings = embeddings.detach().cpu()
        if labels is not None and not labels.device == torch.device("cpu"):
            labels = labels.detach().cpu().numpy()

        if labels is None:
            labels = np.repeat(np.arange(n_classes), n_samples)
        
        data_np = embeddings.numpy()

        # reduce the dimensionality to 2D
        perplex = 9 if n_samples > 9 else 19
        tsne = TSNE(n_components=2, random_state=420, perplexity=perplex)
        data_tsne = tsne.fit_transform(data_np)

        # plot t-SNE with colored points based on classes
        plt.figure(figsize=(8, 6))
        for class_label in range(n_classes):
            mask = (labels == class_label)
            plt.scatter(data_tsne[mask, 0], data_tsne[mask, 1], label=f"class {class_label}")

        ep = f"_epoch_{epoch}" if epoch is not None else ""
        filename = os.path.join(os.getcwd(), "output", f"tsne_epoch{ep}.png")
        plt.title("t-SNE Visualization with Class Colors")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.savefig(filename)
        plt.close()

    @staticmethod
    def visual_attention(img: Tensor, att: Tensor, mean: Optional[List[float]], std: Optional[List[float]], ep: int):
        if att is None:
            return

        # prepare: select a random image, get its size and check if de-normalization is needed
        select: int = torch.randint(0, img.size(0), (1,)).item()
        bs = img.size(0)
        s = img.size(-1)
        denorm = nn.Identity()
        if mean is not None and std is not None:
            denorm = transforms.Normalize(mean=[-m/s for m, s in zip(mean, std)], std=[1/s for s in std])

        # get one image and its corresponding attention
        att = att.detach().cpu()
        img = img.detach().cpu()
        img = denorm(img)

        # resize attention to match the input size and normalize
        att_exp = nn.functional.interpolate(att, size=(s, s), mode='bilinear', align_corners=False)
        min_per_img = att_exp.view(att_exp.size(0), -1).min(dim=1).values.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        max_per_img = att_exp.view(att_exp.size(0), -1).max(dim=1).values.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        att_exp = (att_exp - min_per_img) / (max_per_img - min_per_img + 1e-6)
        heatmap = att_exp.squeeze().numpy()

        # plot with overlay
        rows = int(np.ceil(np.sqrt(bs)))
        cols = int(np.ceil(bs / rows))
        fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(10, 10))

        for i in range(bs):
            row, col = i // cols, i % cols
            axes[row, col].imshow(transforms.ToPILImage()(img[i]))
            axes[row, col].imshow(heatmap[i], cmap="jet", alpha=0.5)
            axes[row, col].axis("off")

        # remove empty subplots
        for i in range(bs, rows * cols):
            row, col = i // cols, i % cols
            fig.delaxes(axes[row, col])

        plt.tight_layout()
        plt.suptitle("Attention Heatmap")
        plt.savefig(os.path.join(os.getcwd(), "output", f"attention_{ep}.png"))
        plt.close()