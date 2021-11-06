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

        self.compiled_head = None
        self.compiled_body = None
        self.compile_model()

    def forward(self, x):
        groups = {g: self.compiled_body[g](x) for g in self.paths}
        out = {}
        for group in groups:
            for task in group:
                out[task] = self.compiled_head[task](groups[group])
        return out

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

    def freeze_model(self):
        for block in self.blocks:
            for layer in self.blocks[block]:
                layer.requires_grad_(requires_grad=False)

    def split_group(self, old_group, new_groups):
        if old_group not in self.groups.values():
            raise KeyError(f"Target group to split \"{old_group}\" either doesn't exist or is already split")
        elif sorted(old_group) != sorted([task for group in new_groups for task in group]):
            raise KeyError(f"There is a mismatch of tasks between the old and new groupings")

        for new_group in new_groups:
            for task in new_group:
                self.groups[task] = new_group

            self.paths[new_group] = self.paths[old_group] + ["_".join(new_group), ]

            self.blocks["_".join(new_group)] = nn.ModuleList()

        del self.paths[old_group]
        self.finished.append("_".join(old_group))

        self.compile_model()

    def compile_model(self):
        self.compiled_body = {group: nn.Sequential(*[nn.Sequential(*self.blocks[block]) for block in self.paths[group]])
                              for group in self.paths}
        self.compiled_head = {task: nn.Sequential(*self.heads[task]) for task in self.heads}

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
