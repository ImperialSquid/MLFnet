class ModelMixin:
    def draw(self, input_tensor, filename=None, filetype="png", transforms="default", **options):
        import hiddenlayer as hl
        import os

        if filename is None:
            filename = self.__class__.__name__

        if options.get("verbose", False):
            transforms = []
            filename = filename + "_Verbose"

        os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'  # make graphviz discoverable
        graph = hl.build_graph(self, input_tensor, transforms=transforms)
        for option in options:
            graph.theme[option] = options[option]
        graph.save(filename, format=filetype)

    def frozen_states(self):
        layers = {}
        for p in self.named_parameters():
            name = ".".join(p[0].split(".")[:-1])
            layers[name] = layers.get(name, []) + [p[1].requires_grad]
        layers = {l: not any(layers[l]) for l in layers}
        return layers
