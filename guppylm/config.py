"""GuppyLM configuration."""

from dataclasses import dataclass


@dataclass
class GuppyConfig:
    vocab_size: int = 4096
    max_seq_len: int = 128
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    ffn_hidden: int = 768
    dropout: float = 0.1

    # Soft Mixture of Experts (Puigcerver et al., 2023)
    use_moe: bool = False
    n_experts: int = 4          # number of expert FFNs
    moe_slots: int = 1          # soft-dispatch slots per expert

    # Recurrent sublayer (minimal GRU over the sequence dimension)
    use_recurrent: bool = False

    # Ouroboros loop — re-apply the full block stack n_loops times (weight-shared)
    use_ouroloop: bool = False
    n_loops: int = 3            # number of ouroboros iterations

    # Special tokens
    pad_id: int = 0
    bos_id: int = 1           # <|im_start|>
    eos_id: int = 2           # <|im_end|>


@dataclass
class TrainConfig:
    batch_size: int = 32
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    warmup_steps: int = 200
    max_steps: int = 10000
    eval_interval: int = 200
    save_interval: int = 500
    grad_clip: float = 1.0
    device: str = "auto"
    seed: int = 42
    data_dir: str = "data"
    output_dir: str = "checkpoints"
