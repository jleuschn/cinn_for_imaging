from copy import deepcopy

from cinn_for_imaging.reconstructors.cinn_reconstructor import CINNReconstructor
from cinn_for_imaging.reconstructors.networks.iunet import IUNet


class IUNetReconstructor(CINNReconstructor):
    """
    Dival reconstructor class for the iUNet network.
    """

    HYPER_PARAMS = deepcopy(CINNReconstructor.HYPER_PARAMS)
    HYPER_PARAMS.update({
        'train_cond': {
            'default': 0.0,
            'retrain': True
        },
        'depth': {
            'default': 4,
            'retrain': True
        },
        'clamp_all': {
            'default': False,
            'retrain': True
        },
        'permute': {
            'default': False,
            'retrain': True
        },
        'permute_type': {
            'default': 'random',
            'retrain': True
        },
        'special_init': {
            'default': True,
            'retrain': True
        },
        'normalize_inn': {
            'default': True,
            'retrain': True
        }})

    def __init__(self, operator, in_ch: int = 1, img_size=None,
                 downsample_levels: int = 5, max_samples_per_run: int = 100,
                 conditioning: str = "fbp",
                 trainer_args: dict = {'distributed_backend': 'ddp',
                                       'gpus': [0]},
                 data_range:list = [0, 1],
                 **kwargs):
        """
        Parameters
        ----------
        ray_trafo : TYPE
            Ray transformation (forward operator).
        in_ch : int, optional
            Number of input channels.
            The default is 1.
        img_size : tuple of int, optional
            Internal size of the reconstructed image. This must be divisible by
            2**i for i=1,...,downsample_levels. By choosing None an optimal
            image size will be determined automatically.
            The default is None.
        max_samples_per_run : int, optional
            Max number of samples for a single run of the network. Adapt to the
            memory of your GPU. To reach the desired samples_per_reco,
            multiple runs of the network will automatically be performed.
            The default is 100.
        trainer_args : dict, optional
            Arguments for the Pytorch Trainer.
            The defaults are distributed_backend='ddp' and gpus=[0]
        Returns
        -------
        None.

        """

        super().__init__(ray_trafo=operator, in_ch=in_ch,
                         img_size=img_size, downsample_levels=downsample_levels,
                         max_samples_per_run=max_samples_per_run,
                         conditioning=conditioning,
                         trainer_args=trainer_args,
                         data_range=data_range,
                         **kwargs)

    def init_model(self):
        """
        Initialize the model.

        Returns
        -------
        None.

        """
        self.model = IUNet(in_ch=self.in_ch,
                           img_size=self.img_size,
                           operator=self.op,
                           sample_distribution=self.sample_distribution,
                           conditioning=self.conditioning,
                           conditional_args={
                               'filter_type': self.filter_type,
                               'frequency_scaling': self.frequency_scaling,
                               'train_cond': self.train_cond},
                           optimizer_args={
                               'lr': self.lr,
                               'weight_decay': self.weight_decay},
                           downsample_levels=self.downsample_levels,
                           clamping=self.clamping,
                           downsampling=self.downsampling,
                           coupling=self.coupling,
                           train_noise=self.train_noise,
                           depth=self.depth,
                           data_range=self.data_range,
                           clamp_all = self.clamp_all,
                           permute = self.permute,
                           special_init = self.special_init,
                           normalize_inn = self.normalize_inn, 
                           permute_type=self.permute_type)
        
    def load_learned_params(self, path, checkpoint:bool = True,
                            strict:bool = False):
        """
        Load a model from the given path. To load a model along with its
        weights, biases and module_arguments use a checkpoint.

        Parameters
        ----------
        path : TYPE
            DESCRIPTION.
        checkpoint : bool, optional
            DESCRIPTION. The default is True.
        strict : bool, optional
            Strict loading of all parameters. Will raise an error if there
            are unknown weights in the file.
            The default is False

        Returns
        -------
        None.

        """
        if checkpoint:
            print("Load from ", path)
            # The operator cannot be stored in the checkpoint file. Therefore,
            # we have to provide him separately.
            self.model = IUNet.load_from_checkpoint(path,
                                                   strict=strict,
                                                   operator=self.op)
            
            # Update the hyperparams of the reconstructor based on the hyper-
            # params of the model. Hyperparams for the optimizer and training
            # routine are ignored.
            hparams = self.model.hparams
            
            # set regular hyperparams
            self.img_size = hparams.img_size
            self.clamping = hparams.clamping
            self.downsample_levels = hparams.downsample_levels
            self.downsampling = hparams.downsampling
            self.sample_distribution = hparams.sample_distribution
            self.coupling = hparams.coupling
            self.conditioning = hparams.conditioning
            self.depth = hparams.depth

            # set hyperparams for the conditional part
            for cond_attr in ['filter_type', 'frequency_scaling']:
                if cond_attr in hparams.conditional_args:
                    self.HYPER_PARAMS[cond_attr] = hparams.conditional_args[
                                                                    cond_attr]
