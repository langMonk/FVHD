from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class FVHDConfig(BaseModel):
    """
    Configuration for FVHD algorithm.
    """

    n_components: int = Field(default=2, description="Number of dimensions for the embedding.")
    nn: int = Field(default=5, description="Number of nearest neighbors.")
    rn: int = Field(default=2, description="Number of random neighbors.")
    c: float = Field(default=0.2, description="Attraction/Repulsion balance coefficient.")
    epochs: int = Field(default=2000, description="Number of optimization epochs.")
    eta: float = Field(default=0.2, description="Learning rate / Step size.")
    device: str = Field(default="cpu", description="Computing device (cpu, cuda, mps).")
    autoadapt: bool = Field(default=True, description="Enable auto-adaptation of learning rate.")
    velocity_limit: bool = Field(default=True, description="Enable velocity limiting.")
    verbose: bool = Field(default=True, description="Enable verbose output.")
    mutual_neighbors_epochs: int | None = Field(
        default=None, description="Epochs to use mutual neighbors graph."
    )
    metric: str = Field(default="euclidean", description="Distance metric for KNN.")
    n_jobs: int = Field(default=-1, description="Number of parallel jobs for KNN.")

    optimizer: Any | None = Field(
        default=None, description="Optimizer class (e.g. torch.optim.Adam)."
    )
    optimizer_kwargs: dict[str, Any] | None = Field(
        default=None, description="Keyword arguments for the optimizer."
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)
