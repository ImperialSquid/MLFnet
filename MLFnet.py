from copy import deepcopy
from typing import Optional, Tuple, Dict

from torch import nn


class MLFnet(nn.Module):
    def __init__(self, tasks: Tuple[str, ...] = tuple(), heads: Optional[Dict[str, nn.Module]] = None):
        super().__init__()
        self.tasks = None
        self.groups = None
        self.paths = None
        self.blocks = None
        self.load_test_setup()

        # self.tasks = tuple(sorted(tasks))
        #
        # self.groups = {task: self.tasks for task in self.tasks}
        #
        # self.paths = {group: ["_".join(group)] for group in self.groups.values()}
        #
        # self.blocks = nn.ModuleDict()
        # self.blocks["_".join(self.tasks)] = nn.ModuleList([nn.Identity(), nn.Identity(), nn.Identity()])

        if heads is None or sorted(heads.keys()) != sorted(self.tasks):
            heads = {task: nn.Flatten() for task in self.tasks}
        self.heads = nn.ModuleDict()
        for task in heads.keys():
            self.heads[task] = heads[task]

        self.compiled_body = None
        self.compiled_head = None
        self.compile_model()

    def forward(self, x):
        return x

    def compile_model(self):
        self.compiled_body = {group: nn.Sequential(*[nn.Sequential(*self.blocks[block])
                                                     for block in self.paths[group]])
                              for group in self.paths}

    def load_test_setup(self):
        self.tasks = ("a", "b", "c")
        self.groups = {"a": ("a", "b"),
                       "b": ("a", "b"),
                       "c": ("c",)}
        self.paths = {("a", "b"): ["a_b_c", "a_b"],
                      ("c",): ["a_b_c", "c"]}
        self.blocks = {"a_b_c": [nn.Identity(), nn.Identity()],
                       "a_b": [nn.Identity(), nn.Identity()],
                       "c": [nn.Identity(), nn.Identity()]}
