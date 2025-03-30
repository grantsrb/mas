import torch
import torch.nn as nn
import numpy as np

class FunctionalComponentAnalysis(nn.Module):
    """
    Functional Component Analysis (FCA) is a method for learning a set of
    orthogonal vectors that can be used to transform data. This is similar
    to PCA, but the vectors are learned in a more general way. The vectors
    are learned by sampling a random vector and orthogonalizing it to all
    previous vectors. This is done by projecting the new vector onto each
    previous vector and subtracting the projection. The new vector is then
    normalized. The process is repeated for each new vector. The vectors
    are stored as parameters in the model.

    The recommended usage is to first train a single component to
    convergence and then freeze it. Then, individually add new
    components and train each while keeping the previous components frozen.
    This process can be repeated to learn as many components as desired.
    """
    def __init__(self,
                 size,
                 max_rank=None,
                 remove_components=False,
                 initialization_vector=None,
                 component_mask=None,
                 means=None,
                 stds=None,
                 orthogonalization_vectors=None,
                 *args, **kwargs):
        """
        Args:
            size: int
                The size of the vectors to be learned.
            max_rank: int or None
                The maximum number of components to learn. If None,
                the maximum rank is equal to the size.
            remove_components: bool
                If True, the model will remove components from
                the vectors in the forward hook.
            initialization_vector: optional tensor
                optionally initialize all new parameters from the
                same initialization_vector
            component_mask: tensor
                optionally argue a mask or index tensor to select
                specific components when constructing the matrix
            means: None or tensor (S,)
                optionally specify a tensor that will be used to
                center the representations before the FCA
            stds: None or tensor (S,)
                optionally specify a tensor that will be used to
                scale the representations before the FCA
            orthogonalization_vectors: None or list-like of tensors [(S,), ...]
                Adds a list of vectors to the list of vectors that are
                excluded from the functional components but are
                included for orthogonality calculations.
        """
        super().__init__()
        # Sample a single, initial vector and normalize it
        self.size = size
        self.max_rank = max_rank if max_rank is not None else size
        self.remove_components = remove_components
        self.initialization_vector = initialization_vector
        self.component_mask = component_mask
        self.set_means(means)
        self.set_stds(stds)
        self.parameters_list = nn.ParameterList([])
        self.train_list = []
        self.fixed_list = []
        self.orthogonalization_mtx = None
        self.add_component()
        self.is_fixed = False
        self.fixed_weight = None
        self.excl_ortho_list = []
        if orthogonalization_vectors is not None:
            self.add_excl_ortho_vectors(orthogonalization_vectors)

    def set_means(self, means):
        self.register_buffer("means", means)

    def set_stds(self, stds):
        self.register_buffer("stds", stds)

    def update_orthogonalization_mtx(self):
        """
        Updates the orthogonalization matrix by concatenating
        the excluded orthogonalization parameters with the
        fixed parameters.
        """
        if len(self.excl_ortho_list)==0 and len(self.fixed_list)==0:
            self.orthogonalization_mtx = []
            self.ortho_cov_mtx = []
            return
        device = self.get_device()
        vecs = [p.data.to(device) for p in self.excl_ortho_list] +\
               [p.data.to(device) for p in self.fixed_list]
        self.orthogonalization_mtx = torch.vstack( vecs )
        self.ortho_cov_mtx = torch.matmul(
            self.orthogonalization_mtx.T, self.orthogonalization_mtx
        )

    def add_excl_ortho_vectors(self, new_vectors):
        """
        Adds a list of vectors to the list of vectors that are
        excluded from the functional components but are
        included for orthogonality calculations.

        Args:
            new_vectors: list of tensors
        """
        for v in new_vectors:
            self.excl_ortho_list.append(v)
        rank_diff = self.size-len(self.excl_ortho_list)
        self.max_rank = min(self.max_rank, rank_diff)
        print("New Max Rank:", self.max_rank)
        self.update_orthogonalization_mtx()

    def add_orthogonalization_vectors(self, new_vectors):
        self.add_excl_ortho_vectors(new_vectors)

    def freeze_parameters(self, freeze=True):
        fixed_list = []
        train_list = []
        for p in self.parameters_list:
            p.requires_grad = not freeze
            if freeze:
                fixed_list.append(p)
            else:
                train_list.append(p)
        self.train_list = train_list
        self.fixed_list = fixed_list
        self.update_orthogonalization_mtx()

    def freeze_weights(self, freeze=True):
        self.freeze_parameters(freeze=freeze)

    def set_fixed(self, fixed=True):
        """
        Can fix the weight matrix in order to quit calculating
        orthogonality via gram schmidt.
        """
        if fixed:
            self.fixed_weight = self.weight
        self.is_fixed = fixed

    def reset_fixed_weight(self,):
        params = self.orthogonalize_parameters()
        matrix = torch.vstack(params)
        self.fixed_weight = matrix

    def orthogonalize_vector(self, new_vector, prev_vectors, prev_is_cov_mtx=False):
        mtx = prev_vectors
        if mtx is not None and len(mtx)>0:
            if type(mtx)==list:
                # Make matrix of previous vectors
                mtx = torch.vstack(mtx)
            # Compute projections
            if prev_is_cov_mtx:
                proj_sum = torch.matmul(mtx, new_vector)
            else:
                proj_sum = torch.matmul(mtx.T, torch.matmul(mtx, new_vector))
            # Subtract projections
            new_vector = new_vector - proj_sum
        # Normalize vector
        new_vector = new_vector / torch.norm(new_vector, 2)
        return new_vector

    def update_parameters(self):
        """
        Orthogonalize all parameters in the list and make a new
        parameter list. Does not track gradients.
        """
        # Sample a new vector and orthogonalize it
        device = self.get_device()
        orth = self.excl_ortho_list
        if len(orth)>0: orth = [o.to(device) for o in orth]
        params = []
        with torch.no_grad():
            for p in self.parameters_list:
                p = self.orthogonalize_vector(
                    p, prev_vectors=orth+params)
                params.append(p)
        self.parameters_list = nn.ParameterList([
            nn.Parameter(p.data) for p in params
        ])
        self.train_list = [p for p in self.parameters_list if p.requires_grad]
        self.fixed_list = [p for p in self.parameters_list if not p.requires_grad]
        self.update_orthogonalization_mtx()

    def orthogonalize_parameters(self):
        """
        Only orthogonalize the parameters that require gradients.
        Does track gradients.
        """
        device = self.get_device()
        if self.orthogonalization_mtx is None:
            self.update_orthogonalization_mtx()
        if len(self.orthogonalization_mtx)>0:
            orth = self.orthogonalization_mtx.to(device)
            cov_mtx = self.ortho_cov_mtx.to(device)
        else:
            orth = []
            cov_mtx = []
        params = []
        for i,p in enumerate(self.parameters_list):
            if p.requires_grad==True:
                if len(params)==0 and len(cov_mtx)>0:
                    p = self.orthogonalize_vector(
                        p,
                        prev_vectors=cov_mtx,
                        prev_is_cov_mtx=True
                    )
                elif len(orth)>0:
                    p = self.orthogonalize_vector(
                        p,
                        prev_vectors=[orth] + params
                    )
                else:
                    p = self.orthogonalize_vector(
                        p,
                        prev_vectors=params
                    )
                params.append(p)
        return self.fixed_list + params

    def make_matrix(self, components=None):
        """
        Create a matrix from the stored parametersa.

        Args:
            components: None or torch LongTensor.
                If None, all components are used. If a
                LongTensor, only the components specified
                by the tensor are used.
        Returns:
            matrix: torch.Tensor.
                The low rank orthogonal matrix created from
                the parameter list.
        """
        if self.is_fixed and self.fixed_weight is not None:
            matrix = self.fixed_weight.to(self.get_device())
        else:
            params = self.orthogonalize_parameters()
            matrix = torch.vstack(params)
        if components is not None:
            matrix = matrix[components]
        elif self.component_mask is not None:
            matrix = matrix[self.component_mask]
        return matrix

    @property
    def weight(self):
        return self.make_matrix()

    @property
    def rank(self):
        return len(self.parameters_list)

    def get_device(self):
        try:
            device = self.parameters_list[0].get_device()
        except: device = -1
        return "cpu" if device<0 else device

    def add_new_axis_parameter(self):
        if len(self.parameters_list) >= self.max_rank:
            return None
        # Sample a new axis and add it to the parameter list
        if self.initialization_vector is not None:
            new_axis = self.initialization_vector.data.clone()
        else:
            new_axis = torch.randn(self.size)
        p = nn.Parameter(new_axis).to(self.get_device())
        self.parameters_list.append(p)
        self.train_list.append(p)
        return self.parameters_list[-1]

    def add_component(self):
        self.add_new_axis_parameter()

    def remove_component(self, idx):
        was_fixed = self.is_fixed
        if self.is_fixed: self.set_fixed(False)
        del_p = self.parameters_list[idx]
        new_list = [p for p in self.parameters_list if p is not del_p]
        self.parameters_list = nn.ParameterList(new_list)
        self.train_list = [p for p in self.train_list if p is not del_p]
        self.fixed_list = [p for p in self.fixed_list if p is not del_p]
        if was_fixed: self.set_fixed(True)

    def load_sd(self, sd):
        """
        Assists in loading a state dict
        """
        n_axes = 0
        for k in sd:
            if "parameters_list" in k:
                ax = int(k.split(".")[-1])
                if ax > n_axes:
                    n_axes = ax
        for _ in range(n_axes+1-self.rank):
            self.add_new_axis_parameter()
        try:
            self.load_state_dict(sd)
        except:
            if "means" in sd and not hasattr(self, "means"):
                self.add_means(sd["means"])
            if "stds" in sd and not hasattr(self, "stds"):
                self.add_means(sd["stds"])
            try:
                self.load_state_dict(sd)
            except:
                print("Failed to load sd")
                print("Current sd:")
                for k in self.state_dict():
                    print(k, self.state_dict()[k].shape)
                print("Argued sd:")
                for k in sd:
                    print(k, sd[k].shape)
                self.load_state_dict(sd)
                assert False

    def init_from_fca(self, fca, freeze_params=True):
        """
        Simplifies starting the parameters from another fca
        object
        """
        self.load_sd(fca.state_dict())
        self.freeze_parameters(freeze=freeze_params)
        self.update_orthogonalization_mtx()
        p = self.add_new_axis_parameter()
        return p
    
    def add_params_from_vector_list(self, vec_list, overwrite=True):
        """
        Adds each of the vectors in the vec_list to the parameters without
        orthogonalizing them.
        
        Args:
            vec_list: list of torch tensors [(S,), ...]
            overwrite: bool
                if true, will overwrite existing parameters before adding new
                ones. Otherwise only initializes new vectors.
        """
        device = self.get_device()
        for i,vec in enumerate(vec_list):
            if overwrite and i<len(self.parameters_list):
                p = self.parameters_list[i]
            else:
                p = self.add_new_axis_parameter()
            p.data = vec.data.clone().to(device)
        self.update_orthogonalization_mtx()

    def get_forward_hook(self):
        def hook(module, input, output):
            fca_vec = self(output)
            stripped = self(fca_vec, inverse=True)
            if self.remove_components:
                stripped = output - stripped
            return stripped
        return hook
    
    def hook_model_layer(self, model, layer):
        for mlayer,modu in model.named_modules():
            if layer==mlayer:
                return modu.register_forward_hook(self.get_forward_hook())
        return None

    def forward(self, x, inverse=False, components=None):
        if inverse:
            x = torch.matmul(
                x, self.make_matrix(components=components)
            )
            if self.stds is not None:
                x = x*self.stds
            if self.means is not None:
                x = x+self.means
            return x
        if self.means is not None:
            x = x-self.means
        if self.stds is not None:
            x = x/self.stds
        return torch.matmul(
            x, self.make_matrix(components=components).T
        )
    
    def interchange_intervention(self, trg, src):
        return trg-self(self(trg),inverse=True)+self(self(src),inverse=True)

def load_fcas_from_path(file_path):
    fca_checkpoint = torch.load(file_path)
    fca_config = fca_checkpoint["config"]
    state_dicts = fca_checkpoint["fca_state_dicts"]
    fcas = {}
    kwargs = fca_config.get("fca_params", {})
    for layer in fca_config["fca_layers"]:
        sd = state_dicts[layer]
        kwargs["size"] = sd[list(sd.keys())[0]].shape[0]
        fcas[layer] = FunctionalComponentAnalysis( **kwargs )
        fcas[layer].load_sd(sd)
        fcas[layer].update_parameters()
        fcas[layer].freeze_parameters()
        fcas[layer].set_fixed(True)
    return fcas

def load_fcas(
        model,
        load_path,
        remove_components=True,
        ret_paths=False,
        verbose=False):
    """
    Simplifies the recursive loading of previous fcas.
    """
    device = "cpu" if model is None else model.get_device()

    # Load Checkpoint
    fca_checkpoint = torch.load(load_path)
    fca_config = fca_checkpoint["config"]

    # Initialize Variables
    fcas = {}
    handles = {}
    loaded_fcas = []
    loaded_handles = []
    loaded_paths = []

    # Recursively Load Previous FCAs
    if fca_config.get("fca_load_path", None) is not None:
        ret = load_fcas(
            model=model,
            load_path=fca_config["fca_load_path"],
            remove_components=remove_components,
            ret_paths=ret_paths,
            verbose=verbose,
        )
        if ret_paths:
            loaded_fcas, loaded_handles, loaded_paths = ret
        else:
            loaded_fcas, loaded_handles = ret
    loaded_paths.append(load_path)

    # Create the FCAs and Load the SDs
    if verbose:
        print("Loading:", load_path)
    state_dicts = fca_checkpoint["fca_state_dicts"]
    kwargs = fca_config.get("fca_params", {})
    modules = {}
    # Attach FCA if model is argued
    if model is not None:
        for layer,modu in model.named_modules():
            modules[layer] = modu
    # Create FCA for each layer
    for layer in state_dicts:
        sd = state_dicts[layer]
        kwargs["size"] = sd[list(sd.keys())[0]].shape[0]
        kwargs["remove_components"] = remove_components
        fcas[layer] = FunctionalComponentAnalysis( **kwargs )
        fcas[layer].load_sd(sd)
        fcas[layer].freeze_parameters()
        fcas[layer].set_fixed(True)
        fcas[layer].to(device)
        if model is not None and layer in modules:
            h = modules[layer].register_forward_hook(
                fcas[layer].get_forward_hook()
            )
            handles[layer] = h

    loaded_handles.append(handles)
    loaded_fcas.append(fcas)
    if ret_paths:
        return loaded_fcas, loaded_handles, loaded_paths
    return loaded_fcas, loaded_handles

def initialize_fcas(
        model,
        config,
        loaded_fcas=[],
        means=None,
        stds=None):
    """
    Args:
        model: torch module
        config: dict
        loaded_fcas: list of FCA objects
        means: dict
            keys: str
                the layer names
            vals: torch tensor (S,)
                the means for that layer
        stds: dict
            keys: str
                the layer names
            vals: torch tensor (S,)
                the stds for that layer
    """
    device = model.get_device()
    fca_handles = []
    fca_parameters = []
    fcas = {}
    handles = {}
    fca_layers = config["fca_layers"]
    kwargs = config.get("fca_params", {})
    for name,modu in model.named_modules():
        if name in fca_layers:
            kwargs["size"] = modu.weight.shape[0]
            kwargs["means"] = means[name]
            kwargs["stds"] = stds[name]
            fcas[name] = FunctionalComponentAnalysis(
                **kwargs
            )
            fcas[name].to(device)
            if config.get("ensure_ortho_chain", False):
                if loaded_fcas:
                    for loaded in loaded_fcas:
                        if name in loaded:
                            print("Loading Orthogonalization Vectors", name)
                            fcas[name].add_excl_ortho_vectors(
                                loaded[name].parameters_list)
            h = modu.register_forward_hook(
                fcas[name].get_forward_hook()
            )
            handles[name] = h
            fca_parameters += list(fcas[name].parameters())
    return fcas, handles, fca_parameters
    

# Example usage
if __name__ == "__main__":

    n_dim = 512
    fca = FunctionalComponentAnalysis(size=n_dim)
    for i in range(n_dim):
        fca.add_new_axis_parameter()

    import time

    prev_list = fca.orthogonalize_parameters()
    prev_list = [p.data for p in prev_list[:-1]]

    print("Base:")
    start = time.time()
    for _ in range(1000):
        base_params = []
        p = fca.parameters_list[-1]
        p = fca.orthogonalize_vector(p, prev_vectors=prev_list)
        base_params.append(p)
    print("Time:", time.time()-start)

    mtx = torch.vstack([v.data for v in prev_list])
    print("Fast:")
    start = time.time()
    for _ in range(1000):
        fast_params = []
        p = fca.parameters_list[-1]
        p = fca.orthogonalize_vector(p, prev_vectors=mtx)
        fast_params.append(p)
    print("Time:", time.time()-start)

    cov = torch.matmul(mtx.T, mtx)
    print("Cov:")
    start = time.time()
    for _ in range(1000):
        cov_params = []
        p = fca.parameters_list[-1]
        p = fca.orthogonalize_vector(p, prev_vectors=cov, prev_is_cov_mtx=True)
        cov_params.append(p)
    print("Time:", time.time()-start)

    for bp,fp,cv in zip(base_params, fast_params, cov_params):
        mse = ((bp-fp)**2).mean()
        if mse>1e-6:
            print("BP:", bp[:5])
            print("FP:", fp[:5])
            print("MSE:", mse)
            print()
        mse = ((bp-cv)**2).mean()
        if mse>1e-6:
            print("BP:", bp[:5])
            print("Cv:", cv[:5])
            print("MSE:", mse)
            print()
        break

    print("End Fast Comparison")


    # Assert that the vectors are orthogonal
    fca.update_parameters()
    for i in range(len(fca.parameters_list)):
        for j in range(i):
            assert torch.dot(fca.parameters_list[i], fca.parameters_list[j]) < 1e-6
    print("Rank:", fca.rank)
    vec = torch.randn(1,n_dim)
    mtx = fca.weight[:3]
    rot = torch.matmul(vec, mtx.T)
    new_vec = torch.matmul(rot, mtx)
    diff = vec - new_vec
    zero = torch.matmul(diff, mtx.T)

    print("Old Vec:", vec)
    print("New Vec:", new_vec)
    print("Diff:", diff)
    print("Zero:", zero)
    print("MSE:", ((vec-new_vec)**2).sum())
    print()

    mtx = fca.weight[3:]
    rot = torch.matmul(vec, mtx.T)
    nnew_vec = torch.matmul(rot, mtx)
    ddiff = vec - nnew_vec
    zzero = torch.matmul(ddiff, mtx.T)

    print("Old Vec:", vec)
    print("New Vec:", nnew_vec)
    print("Diff:", ddiff)
    print("Zero:", zzero)
    print("MSE:", ((vec-new_vec-nnew_vec)**2).sum())
    print()
    
    rot = fca(vec)
    new_vec = fca(rot, inverse=True)
    print("Old Vec:", vec)
    print("New Vec:", new_vec)
    print("MSE:", ((vec-new_vec)**2).sum())
