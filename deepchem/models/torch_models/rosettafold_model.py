"""DeepChem TorchModel wrapper for RosettaFold-style denoising.

This module provides a ``TorchModel`` integration around the
``RosettaFoldDenoiser`` module for training and inference in DeepChem
workflows.
"""

from typing import Optional

from deepchem.models.losses import L2Loss
from deepchem.models.torch_models.rosettafold import RosettaFoldDenoiser
from deepchem.models.torch_models.rosettafold_config import RosettaFoldConfig
from deepchem.models.torch_models.torch_model import TorchModel


class RosettaFoldModel(TorchModel):
    """DeepChem wrapper around a RosettaFold-style denoiser network.

    The underlying module follows the RFdiffusion denoiser contract
    ``model([x_t, t]) -> predicted_noise`` and can therefore be used for
    supervised denoising objectives with standard L2 loss.

    Parameters
    ----------
    config : RosettaFoldConfig, optional
        Configuration object used to build the underlying
        :class:`RosettaFoldDenoiser`.
    batch_size : int, default=8
        Default batch size used by inherited training and prediction methods.
    model_dir : str, optional
        Directory for checkpoints and metadata.
    learning_rate : float, default=1e-4
        Optimizer learning rate.
    **kwargs
        Additional keyword arguments forwarded to :class:`TorchModel`.

    Examples
    --------
    >>> import numpy as np
    >>> import deepchem as dc
    >>> from deepchem.models.torch_models import RosettaFoldModel
    >>> model = RosettaFoldModel()
    >>> x = np.random.randn(2, 16, 9).astype("float32")
    >>> t = np.random.randint(0, 1000, size=(2,)).astype("float32")
    >>> y = np.random.randn(2, 16, 9).astype("float32")
    >>> dataset = dc.data.NumpyDataset(X=np.stack([x, np.broadcast_to(t[:, None, None], x.shape)], axis=1), y=y)

    Notes
    -----
    For complete diffusion workflows, users commonly call this model via custom
    generators that emit multi-input batches in the form ``([x_t, t], [noise],
    [weights])``.
    """

    def __init__(self,
                 config: Optional[RosettaFoldConfig] = None,
                 batch_size: int = 8,
                 model_dir: Optional[str] = None,
                 learning_rate: float = 1e-4,
                 **kwargs) -> None:
        model = RosettaFoldDenoiser(config=config)
        super().__init__(model=model,
                         loss=L2Loss(),
                         output_types=["prediction"],
                         batch_size=batch_size,
                         model_dir=model_dir,
                         learning_rate=learning_rate,
                         **kwargs)
