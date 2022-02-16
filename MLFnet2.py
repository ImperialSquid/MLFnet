from typing import Optional, Tuple, Dict, Type

from torch import nn

from utils import ModelMixin


class MLFnet(nn.Module, ModelMixin):
    def __init__(self, tasks: Tuple[str, ...] = tuple(), heads: Optional[Dict[str, Type[nn.Module]]] = None,
                 backbone: Optional[Type[nn.Module]] = None, device=None, debug=False):
        super().__init__()

    def forward(self, x):
        pass

    def add_layer(self, target_group: Optional[Tuple[str, ...]] = None, **kwargs):
        pass

    def freeze_model(self, target_group: Optional[Tuple[str, ...]] = None):
        pass

    def split_group(self, old_group: Tuple[str, ...], new_groups: Tuple[Tuple[str, ...], ...]):
        pass

    def frozen_states(self):
        pass

    def collect_weight_updates(self, losses: Dict[str, Type[nn.Module]]):
        pass

    def reset_heads(self, target_tasks: Optional[Tuple[str, ...]] = None):
        pass

    def _layer_from_dict(self, layer_dict: Dict[str, str] = None):
        pass

    def compile_model(self):
        pass


def main():
    pass


if __name__ == "__main__":
    main()
