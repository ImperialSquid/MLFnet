from importlib import import_module
from typing import Optional, Tuple, Dict, Type

from torch import nn, concat
from torch.nn.utils import parameters_to_vector

from utils import ModelMixin


# TODO Publish: formalised javadoc style documentation
# TODO Publish: finish and double check type checking
class MLFnet(nn.Module, ModelMixin):
    def __init__(self, tasks: Tuple[str, ...] = tuple(), heads: Optional[Dict[str, Type[nn.Module]]] = None,
                 backbone: Optional[Type[nn.Module]] = None, device=None, debug=False):
        super().__init__()
        if debug:
            self.load_test_setup()

        self.device = device

        if backbone is None:
            backbone = nn.Identity()  # if no backbone is given just use an Identity layer to change nothing
        self.backbone = backbone

        self.tasks = tuple(sorted(tasks))  # stores a tuple of the task names
        self.groups = {task: self.tasks for task in self.tasks}  # maps task names to the group they belong to
        # maps group tuples to a list of the blocks they pass through (all tasks in a group share the same
        # path but with separate heads)
        self.paths = {group: ["_".join(group)] for group in self.groups.values()}

        # blocks maps block names to ModuleLists (names are derived from the names of the tasks that pass through them)
        # ModuleDict and  ModuleList are used over builtins since they make PyTorch aware of any Module's existence
        self.blocks = nn.ModuleDict()
        self.blocks["_".join(self.tasks)] = nn.ModuleList()
        self.finished = []  # any block that has subsequent blocks is "finished" and shouldn't be added to

        # default to an Identity head for any not provided
        self.heads = {task: nn.ModuleList([nn.Identity()]) for task in self.tasks}

        if heads is not None:
            for task in self.tasks:
                if task in heads.keys():  # if a task is given a head, reassign to that instead
                    self.heads[task] = heads[task]

        # the compiled attrs store versions where the lists have been nn.Sequential-ised for simplicity later
        self.compiled_head = None
        self.compiled_blocks = None
        self.compile_model()

    def forward(self, x):
        x = self.backbone(x)  # pass data through shared backbone before heads

        # group_results stores the output for each group's path before heads
        group_results = {}
        # block results stores the output of each block
        block_results = {}
        # since each block sees exactly one input it's safe to cache the results
        # this also saves on multiple passes for expensive blocks
        for group in self.paths:  # iter over each group so we take every possible path through
            group_results[group] = x
            # iter over each block in a path until they have all been applied (or fetched from cache)
            for block in self.paths[group]:
                if block in block_results:
                    group_results[group] = block_results[block]
                else:
                    result = self.compiled_blocks[block](group_results[group])
                    # update group results as we go through, this will be overwritten until the last block
                    group_results[group] = result
                    block_results[block] = result

        out = {}
        # now take each group result and pass it through it's associated head
        for group in group_results:
            for task in group:
                out[task] = self.compiled_head[task](group_results[group])
        return out

    def add_layer(self, target_group: Optional[Tuple[str, ...]] = None, **kwargs):
        if target_group is not None:
            target_block = "_".join(sorted(target_group))
            if target_block not in self.blocks or target_block in self.finished:  # more helpful error to raise
                raise KeyError(f"Target block \"{target_block}\" either doesn't exist or is finished training")

        layer = getattr(import_module("torch.nn"), kwargs["type"])  # import and instantiate layers on the fly
        layer_kwargs = {kw: kwargs[kw] for kw in kwargs if kw != "type"}

        # need to keep track of newly added layers so they can be passed back and put into the optimiser
        new_layers = []

        if target_group is not None:  # add the new layer to the desired group
            new_layer = layer(**layer_kwargs)
            new_layer.to(self.device)
            self.blocks[target_block].append(new_layer)
            new_layers.append(new_layer)
        else:  # None is allowed when adding layers to every group (ie every block not in finished)
            for unfinished in [block for block in self.blocks if block not in self.finished]:
                # new layers are instantiated inside the loop to prevent references to the same layer
                new_layer = layer(**layer_kwargs)
                new_layer.to(self.device)
                self.blocks[unfinished].append(new_layer)
                new_layers.append(new_layer)

        # TODO Investigate: look into whether forcing a reset_head is always needed, this may be a temporary line and
        #  it's up to the user to choose in later versions
        self.reset_heads(target_tasks=target_group)  # we don't want old heads affecting new layers so they are reset
        self.compile_model()  # recompile model to update the Sequentials

        return new_layers

    def freeze_model(self, target_group: Optional[Tuple[str, ...]] = None):
        # freeze every block in the specified group's path
        for block in [b for b in self.blocks if b in self.paths[target_group]]:
            # PyCharm doesn't like self.blocks[block] here since it doesn't see ModuleLists as iterable (they are)
            for layer in self.blocks[block]:
                layer.requires_grad_(requires_grad=False)

    def split_group(self, old_group, new_groups):
        if old_group not in self.groups.values():  # do some checks to see if the group is legal
            raise KeyError(f"Target group to split \"{old_group}\" either doesn't exist or is already split")
        elif sorted(old_group) != sorted([task for group in new_groups for task in group]):
            raise KeyError(f"There is a mismatch of tasks between the old and new groupings")

        for new_group in new_groups:
            for task in new_group:
                self.groups[task] = new_group  # reassign tasks to their new group
            # update the path by adding the new block name
            self.paths[new_group] = self.paths[old_group] + ["_".join(new_group), ]
            self.blocks["_".join(new_group)] = nn.ModuleList()  # create new block

        del self.paths[old_group]  # make sure to delete the old group's path
        self.finished.append("_".join(old_group))  # mark the block as finished so no more layers are added

        self.compile_model()  # recompile model

    def frozen_states(self):
        layers = {}
        for p in self.named_parameters():
            name = ".".join(p[0].split(".")[:-1])
            layers[name] = layers.get(name, []) + [p[1].requires_grad]
        layers = {l: not any(layers[l]) for l in layers}
        return layers

    def assess_grouping(self, losses: Dict[str, nn.Module], method: str = "", **kwargs):
        # TODO Extras: enforce only checking in task groups or super groups (eg if tasks are [(a,b),(c)], allow (a,b),
        #  (c) and (a,b,c))
        frozen_states = self.frozen_states()
        vectors = {}
        self.zero_grad()
        for task in losses:
            losses[task].backwards()

            modules = []
            for layer in [layer for layer in sorted(frozen_states.keys()) if not frozen_states[layer]]:
                modules.append(parameters_to_vector(layer))
            vectors[task] = concat(modules)

            self.zero_grad()

        # TODO Publish: check self._assess_grouping_<method> exists, throw AttributionError otherwise
        # TODO Publish: use inspect.signature to check signature (accepts vectors, kwargs and none other),
        #  throw AttributionError otherwise
        comparison_method = getattr(self, "_assess_grouping_" + method)

        grouping = comparison_method(vectors=vectors, **kwargs)
        return grouping

    # TODO Extras: add a version that tests all numbers of clusters and returns a dict?
    def _assess_grouping_kmeans(self, vectors, **kwargs):
        from sklearn.cluster import KMeans

        groups = KMeans(**kwargs).fit(vectors)

        if kwargs["debug"]:
            print(f"{vectors=}")
            print(f"{groups=}")

        return groups

    # TODO Extras: add a version that tests all numbers of clusters and returns a dict?
    def _assess_grouping_dbscan(self, vectors, **kwargs):
        from sklearn.cluster import DBSCAN

        groups = DBSCAN(**kwargs).fit(vectors)

        if kwargs["debug"]:
            print(f"{vectors=}")
            print(f"{groups=}")

        return groups

    def _assess_grouping_agglomerative_clustering(self, vectors, **kwargs):
        from sklearn.cluster import FeatureAgglomeration

        groups = FeatureAgglomeration(**kwargs).fit(vectors)

        if kwargs["debug"]:
            print(f"{vectors=}")
            print(f"{groups=}")

        return groups

    def reset_heads(self, target_tasks: Optional[Tuple[str, ...]] = None):
        if target_tasks is None:
            target_tasks = self.tasks

        for task in target_tasks:
            for layer in self.heads[task]:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

    def compile_model(self):
        # safe to use dict comprehension here since PyTorch is already aware of the Modules
        self.compiled_blocks = {block: nn.Sequential(*self.blocks[block]) for block in self.blocks}
        self.compiled_head = {task: nn.Sequential(*self.heads[task]) for task in self.heads}

    def load_test_setup(self):
        # just an example setup
        self.tasks = ("a", "b", "c")
        self.groups = {"a": ("a", "b"),
                       "b": ("a", "b"),
                       "c": ("c",)}
        self.paths = {("a", "b"): ["a_b_c", "a_b"],
                      ("c",): ["a_b_c", "c"]}
        self.blocks = {"a_b_c": nn.ModuleList([nn.Flatten(), nn.Flatten()]),
                       "a_b": nn.ModuleList([nn.Flatten(), nn.Flatten(), nn.Flatten(), nn.Flatten()]),
                       "c": nn.ModuleList([nn.Flatten(), nn.Flatten(), nn.Flatten()])}
        self.compiled_head = None
        self.compiled_blocks = None
        self.compile_model()


def example_case(case):
    import torch
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # use GPU if CUDA is available
    print(device)
    if case in ["normal", "unequal"]:
        model = MLFnet(tasks=("a", "b", "c"), heads=None)
        model.add_layer(target_group=None,
                        **{"type": "Conv2d", "in_channels": 3, "out_channels": 128, "kernel_size": (3, 3)})
        model.add_layer(target_group=None,
                        **{"type": "Conv2d", "in_channels": 128, "out_channels": 256, "kernel_size": (3, 3)})
        model.add_layer(target_group=None,
                        **{"type": "Conv2d", "in_channels": 256, "out_channels": 512, "kernel_size": (3, 3)})

        model.split_group(old_group=("a", "b", "c"), new_groups=[("a", "b"), ("c",)])
        model.add_layer(target_group=None,
                        **{"type": "Conv2d", "in_channels": 512, "out_channels": 1024, "kernel_size": (3, 3)})
        model.add_layer(target_group=None,
                        **{"type": "Conv2d", "in_channels": 1024, "out_channels": 1024, "kernel_size": (3, 3)})

        if case == "normal":
            model.add_layer(target_group=None,
                            **{"type": "Conv2d", "in_channels": 1024, "out_channels": 2048, "kernel_size": (3, 3)})
            model.add_layer(target_group=None,
                            **{"type": "Conv2d", "in_channels": 2048, "out_channels": 4096, "kernel_size": (3, 3)})
            print(model)
            model.draw(torch.zeros(1, 3, 96, 96), filename="architectures/MLFnet")
            model.draw(torch.zeros(1, 3, 96, 96), filename="architectures/MLFnet", verbose=True)
        elif case == "unequal":
            model.add_layer(target_group=("a", "b"),
                            **{"type": "Conv2d", "in_channels": 1024, "out_channels": 2048, "kernel_size": (3, 3)})
            model.add_layer(target_group=("a", "b"),
                            **{"type": "Conv2d", "in_channels": 2048, "out_channels": 4096, "kernel_size": (3, 3)})
            print(model)
            model.draw(torch.zeros(1, 3, 96, 96), filename="architectures/MLFnetUnequal")
            model.draw(torch.zeros(1, 3, 96, 96), filename="architectures/MLFnetUnequal", verbose=True)
    elif case == "many":
        import string
        tasks = tuple(string.ascii_letters[:10])
        model = MLFnet(tasks=tasks, heads=None)
        model.add_layer(target_group=None,
                        **{"type": "Conv2d", "in_channels": 3, "out_channels": 3, "kernel_size": (3, 3)})

        model.split_group(old_group=tasks, new_groups=[tasks[:4], tasks[4:]])
        model.add_layer(target_group=None,
                        **{"type": "Conv2d", "in_channels": 3, "out_channels": 3, "kernel_size": (3, 3)})

        model.split_group(old_group=tasks[:4], new_groups=[tasks[:2], tasks[2:4]])
        model.add_layer(target_group=tasks[:2],
                        **{"type": "Conv2d", "in_channels": 3, "out_channels": 3, "kernel_size": (3, 3)})

        model.split_group(old_group=tasks[4:], new_groups=[tasks[4:7], tasks[7:]])
        model.add_layer(target_group=tasks[4:7],
                        **{"type": "Conv2d", "in_channels": 3, "out_channels": 3, "kernel_size": (3, 3)})
        print(model)

        model.draw(torch.zeros(1, 3, 96, 96), filename="architectures/MLFnetMany")
        model.draw(torch.zeros(1, 3, 96, 96), filename="architectures/MLFnetMany", verbose=True)


def main():
    pass


if __name__ == "__main__":
    main()
