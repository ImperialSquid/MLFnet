from importlib import import_module
from typing import Optional, Tuple, Dict

from torch import nn

from utils import ModelMixin


class MLFnet(nn.Module, ModelMixin):
    def __init__(self, tasks: Tuple[str, ...] = tuple(), heads: Optional[Dict[str, nn.Module]] = None):
        super().__init__()
        self.tasks = tuple(sorted(tasks))
        self.groups = {task: self.tasks for task in self.tasks}
        self.paths = {group: ["_".join(group)] for group in self.groups.values()}

        self.blocks = nn.ModuleDict()
        self.blocks["_".join(self.tasks)] = nn.ModuleList()
        self.finished = []

        if heads is None or sorted(heads.keys()) != sorted(self.tasks):
            heads = {task: nn.ModuleList([nn.Flatten()]) for task in self.tasks}
        self.heads = nn.ModuleDict()
        for task in heads.keys():
            self.heads[task] = heads[task]

        self.compiled = None
        self.compile_model()

    def forward(self, x):
        x = {task: self.compiled[task](x) for task in self.groups}
        return x

    def add_layer(self, target_group: Optional[Tuple[str, ...]] = None, **kwargs):
        if target_group is not None:
            target_block = "_".join(sorted(target_group))
            if target_block not in self.blocks or target_block in self.finished:
                raise KeyError(f"Target block \"{target_block}\" either doesn't exist or is finished training")

        new_layer = getattr(import_module("torch.nn"), kwargs["type"])  # import and instantiate layers on the fly
        layer_kwargs = {kw: kwargs[kw] for kw in kwargs if kw != "type"}

        if target_group is not None:
            self.blocks[target_block].append(new_layer(**layer_kwargs))
        else:
            for unfinished in [block for block in self.blocks if block not in self.finished]:
                self.blocks[unfinished].append(new_layer(**layer_kwargs))

        self.compile_model()

    def compile_model(self):
        compiled_body = {group: nn.Sequential(*[nn.Sequential(*self.blocks[block]) for block in self.paths[group]])
                         for group in self.paths}
        compiled_head = {task: nn.Sequential(*self.heads[task]) for task in self.heads}

        self.compiled = {task: nn.Sequential(*[compiled_body[self.groups[task]], compiled_head[task]])
                         for task in self.groups}

    def load_test_setup(self):
        self.tasks = ("a", "b", "c")
        self.groups = {"a": ("a", "b"),
                       "b": ("a", "b"),
                       "c": ("c",)}
        self.paths = {("a", "b"): ["a_b_c", "a_b"],
                      ("c",): ["a_b_c", "c"]}
        self.blocks = {"a_b_c": nn.ModuleList([nn.Flatten(), nn.Flatten()]),
                       "a_b": nn.ModuleList([nn.Flatten(), nn.Flatten(), nn.Flatten(), nn.Flatten()]),
                       "c": nn.ModuleList([nn.Flatten(), nn.Flatten(), nn.Flatten()])}
