from importlib import import_module
from inspect import signature
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

        # blocks maps block names to lists of layers
        # (names are derived from the names of the tasks that pass through them)
        self.blocks = dict()
        self.blocks["_".join(self.tasks)] = list()
        self.finished = []  # any block that has subsequent blocks is "finished" and shouldn't be added to

        # default to an Identity head for any not provided
        self.heads = dict()
        for task in tasks:
            self.heads[task] = [nn.Identity().to(self.device)]

        if heads is not None:
            for task in self.tasks:
                if task in heads.keys():  # if a task is given a head, reassign to that instead
                    self.heads[task] = heads[task].to(self.device)

        self.update_vectors = dict()

        # the compiled attrs store versions where the lists have been nn.Sequential-ised for simplicity later
        # ModuleDict and  ModuleList are used over builtins since they make PyTorch aware of any Module's existence
        self._compiled_heads = nn.ModuleDict()
        self._compiled_blocks = nn.ModuleDict()
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
                    result = self._compiled_blocks[block](group_results[group])
                    # update group results as we go through, this will be overwritten until the last block
                    group_results[group] = result
                    block_results[block] = result

        out = {}
        # now take each group result and pass it through it's associated head
        for group in group_results:
            for task in group:
                out[task] = self._compiled_heads[task](group_results[group])
        return out

    def add_layer(self, target_group: Optional[Tuple[str, ...]] = None, **kwargs):
        if target_group is not None:
            target_block = "_".join(sorted(target_group))  # derive block name by joining task names with underscores
            if target_block not in self.blocks or target_block in self.finished:  # more helpful error to raise
                raise KeyError(f"Target block \"{target_block}\" either doesn't exist or is finished training (blocks "
                               f"halfway along a path are considered finished to ensure consistent behaviour in "
                               f"later blocks)")
        if kwargs["type"] == "custom" and target_group is None:
            raise ValueError("Cannot add custom blocks when target_group is None. Custom blocks are given already "
                             "instantiated, adding to multiple groups would effectively recombine them.")

        # need to keep track of newly added layers so they can be passed back and put into the optimiser
        # optimiser won't update layers it doesn't know about so the results of this should be passed to
        # optimiser.add_param_group()
        new_layers = []

        if kwargs["type"] == "custom":  # add custom layers through "custom" type
            # this is aimed at more complex Seqentials of layers which are to be repeated in future
            # WARNING passing the same object in multiple times will get around regrouping checks,
            # this is on the user to ensure
            # TODO Extras: safety checks that new layer is a proper type to be added
            # TODO Extras: consider avoiding issue of accidental regroups using deepcopy?
            new_layer = kwargs["custom"].to(self.device)  # get custom block and move to device
            self.blocks[target_block].append(new_layer)
            new_layers.append(new_layer)  # new block should be returned as usual
        else:
            layer = getattr(import_module("torch.nn"), kwargs["type"])  # import and instantiate layers on the fly
            layer_kwargs = {kw: kwargs[kw] for kw in kwargs if kw != "type"}

            if target_group is not None:  # add the new layer to the desired group
                new_layer = layer(**layer_kwargs).to(self.device)
                self.blocks[target_block].append(new_layer)
                new_layers.append(new_layer)
            else:  # None is allowed when adding layers to every group (ie every block not in finished)
                for unfinished in [block for block in self.blocks if block not in self.finished]:
                    # new layers are instantiated inside the loop to prevent references to the same layer
                    new_layer = layer(**layer_kwargs).to(self.device)
                    self.blocks[unfinished].append(new_layer)
                    new_layers.append(new_layer)

        # TODO Investigate: look into whether forcing a reset_head is always needed, this may be a temporary line and
        #  it's up to the user to choose in later versions
        self.reset_heads(target_tasks=target_group)  # we don't want old heads affecting new layers so they are reset
        self.compile_model()  # recompile model to update the Sequentials

        return new_layers

    def freeze_model(self, target_group: Optional[Tuple[str, ...]] = None):
        # freeze every block in the specified group's path
        for block in [b for b in self.blocks if target_group is None or b in self.paths[target_group]]:
            for layer in self.blocks[block]:
                layer.requires_grad_(requires_grad=False)

    def split_group(self, old_group: Tuple[str, ...], new_groups: Tuple[Tuple[str, ...], ...]):
        if old_group not in self.groups.values():  # do some checks to see if the group is legal
            raise KeyError(f"Target group to split \"{old_group}\" either doesn't exist or is already split")
        elif sorted(old_group) != sorted([task for group in new_groups for task in group]):
            raise KeyError(f"There is a mismatch of tasks between the old and new groupings")

        for new_group in new_groups:
            new_group = sorted(new_group)
            for task in new_group:
                self.groups[task] = new_group  # reassign tasks to their new group
            # update the path by adding the new block name
            self.paths[new_group] = self.paths[old_group] + ["_".join(new_group), ]
            self.blocks["_".join(new_group)] = list()  # create new block

        del self.paths[old_group]  # make sure to delete the old group's path
        self.finished.append("_".join(old_group))  # mark the block as finished so no more layers are added

        self.compile_model()  # recompile model

    def frozen_states(self):  # fetches a dict of layers and whether they are frozen (ie is .requires_grad True?)
        layers = {}
        for p in self.named_parameters():  # iter over layers
            name = ".".join(p[0].split(".")[:-1])  # extract out the name
            layers[name] = layers.get(name, []) + [p[1].requires_grad]
        layers = {l: not any(layers[l]) for l in layers}
        return layers

    def collect_weight_updates(self, losses: Dict[str, Type[nn.Module]]):
        self.zero_grad()
        for task in losses:
            # retain_graph is required since we are backward-ing multiple losses over the same layers separately
            losses[task].backward(retain_graph=True)

            modules = []
            for name, param in self.named_parameters():
                if param.requires_grad and "blocks" in name:  # filter frozen params and those not in the blocks
                    modules.append(parameters_to_vector(param.grad))
            self.update_vectors[task] = self.update_vectors.get(task, list()), + concat(modules).tolist()

            self.zero_grad()

    def assess_grouping(self, group: Tuple[str, ...], method: str = "", **kwargs):
        # TODO Extras: allow "auto" as method, look at number of losses given and go from there?
        # TODO Extras: Add optional preprocessing (eg PCA?)
        if tuple(sorted(group)) not in self.groups.values():
            raise ValueError(f"Given group {group} is not appropriate for assessing. Can only assess "
                             f"currently grouped tasks.")
        if method == "":
            good_methods = ", ".join([func.replace("_assess_grouping_", "") for func in dir(self)
                                      if func.startswith("_assess_grouping_")])
            raise ValueError(f"Grouping method not provided, must be one of: {good_methods}")

        comparison_method = getattr(self, "_assess_grouping_" + method, None)
        if comparison_method is None:  # check grouping method exists, provide useful error if not
            good_methods = ", ".join([func.replace("_assess_grouping_", "") for func in dir(self)
                                      if func.startswith("_assess_grouping_")])
            raise AttributeError(f"Grouping method {method} doesn't exist, use one of: {good_methods}")
        if str(signature(comparison_method)) != "(vectors, **kwargs)":
            # check signature will work if user has implemented their own method
            raise AttributeError("Grouping method signature looks wrong (should be (vectors, **kwargs))")

        raise NotImplementedError("Have not finished reformatting code to allow for grouping assessment")
        # grouping = comparison_method(vectors= XYZ , **kwargs)
        # TODO reformat built-ins to return groups in a manner that can be fed straight back in (0,0,1) -> ((a,b),(c,))
        # self.update_vectors = dict()
        # return grouping

    # TODO Extras: add a version that tests all numbers of clusters and returns a dict?
    def _assess_grouping_kmeans(self, vectors, **kwargs):
        # use kmeans from scikit-learn, options are left to the use to define
        from sklearn.cluster import KMeans

        debug = kwargs["debug"]
        kwargs.pop("debug")

        vs = [vectors[key] for key in vectors]
        groups = KMeans(**kwargs).fit(vs).labels_

        if debug:
            print("Vectors:\n" + "\n".join([key + "\t" + str(vectors[key]) for key in vectors]))
            print(f"{groups=}")
        return groups

    # TODO Extras: add a version that tests all numbers of clusters and returns a dict?
    def _assess_grouping_dbscan(self, vectors, **kwargs):
        # use DBSCAN from scikit-learn, options are left to the use to define
        from sklearn.cluster import DBSCAN

        debug = kwargs["debug"]
        kwargs.pop("debug")

        vs = [vectors[key] for key in vectors]
        groups = DBSCAN(**kwargs).fit(vs).labels_

        if debug:
            print("Vectors:\n" + "\n".join([key + "\t" + str(vectors[key]) for key in vectors]))
            print(f"{groups=}")
        return groups

    def _assess_grouping_agglomerative_clustering(self, vectors, **kwargs):
        # use hierarchical clustering from scikit-learn, options are left to the user to define
        from sklearn.cluster import AgglomerativeClustering

        debug = kwargs["debug"]
        kwargs.pop("debug")

        vs = [vectors[key] for key in vectors]
        groups = AgglomerativeClustering(**kwargs).fit(vs).labels_

        if debug:
            print("Vectors:\n" + "\n".join([key + "\t" + str(vectors[key]) for key in vectors]))
            print(f"{groups=}")
        return groups

    def reset_heads(self, target_tasks: Optional[Tuple[str, ...]] = None):
        if target_tasks is None:  # if tasks is None, reset all
            target_tasks = self.tasks

        for task in target_tasks:
            for layer in self.heads[task]:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()  # use built-in reset_parameter from pytorch

    def compile_model(self):
        # safe to use dict comprehension here since PyTorch is already aware of the layers
        # compiling has no effect on running speed, it just improves code readability elsewhere
        for block in self.blocks:
            self._compiled_blocks[block] = nn.Sequential(*self.blocks[block])
        for head in self.heads:
            self._compiled_heads[head] = nn.Sequential(*self.heads[head])

    def load_test_setup(self):
        # just an example setup
        self.tasks = ("a", "b", "c")
        self.groups = {"a": ("a", "b"),
                       "b": ("a", "b"),
                       "c": ("c",)}
        self.paths = {("a", "b"): ["a_b_c", "a_b"],
                      ("c",): ["a_b_c", "c"]}
        self.blocks = {"a_b_c": [nn.Flatten(), nn.Flatten()],
                       "a_b": [nn.Flatten(), nn.Flatten(), nn.Flatten(), nn.Flatten()],
                       "c": [nn.Flatten(), nn.Flatten(), nn.Flatten()]}
        self._compiled_heads = nn.ModuleDict()
        self._compiled_blocks = nn.ModuleDict()
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

        model.split_group(old_group=("a", "b", "c"), new_groups=(("a", "b"), ("c",)))
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

        model.split_group(old_group=tasks, new_groups=(tasks[:4], tasks[4:]))
        model.add_layer(target_group=None,
                        **{"type": "Conv2d", "in_channels": 3, "out_channels": 3, "kernel_size": (3, 3)})

        model.split_group(old_group=tasks[:4], new_groups=(tasks[:2], tasks[2:4]))
        model.add_layer(target_group=tasks[:2],
                        **{"type": "Conv2d", "in_channels": 3, "out_channels": 3, "kernel_size": (3, 3)})

        model.split_group(old_group=tasks[4:], new_groups=(tasks[4:7], tasks[7:]))
        model.add_layer(target_group=tasks[4:7],
                        **{"type": "Conv2d", "in_channels": 3, "out_channels": 3, "kernel_size": (3, 3)})
        print(model)

        model.draw(torch.zeros(1, 3, 96, 96), filename="architectures/MLFnetMany")
        model.draw(torch.zeros(1, 3, 96, 96), filename="architectures/MLFnetMany", verbose=True)


def main():
    pass


if __name__ == "__main__":
    main()
