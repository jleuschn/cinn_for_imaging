import pytorch_lightning as pl 
from torch.utils.data import DataLoader
from dival import get_standard_dataset

class EllipsesDataModule(pl.LightningDataModule):

    def __init__(self, batch_size:int=4, 
                    impl:str='astra_cuda', 
                    num_data_loader_workers:int=0):
        """
        Initialize the data module for the Ellipses dataset

        Parameters
        ----------
        batch_size : int, optional
            Size of a mini batch.
            The default is 4.
        impl : str, optional
            Which ASTRA implementation should be used.
            The default is 'astra_cuda'.
        num_data_loader_workers : int, optional
            Number of workers.
            The default is 0.

        Returns
        -------
        None.

        """
        super().__init__()
        
        self.batch_size = batch_size
        self.impl = impl 
        self.num_data_loader_workers = num_data_loader_workers
        self.data_range = [0, 1]

    def prepare_data(self):
        """
        Preparation steps like downloading etc. 
        Don't use self here!

        Returns
        -------
        None.

        """
        None

    def setup(self, stage:str = None):
        """
        This is called by every GPU. Self can be used in this context!

        Parameters
        ----------
        stage : str, optional
            Current stage, e.g. 'fit' or 'test'.
            The default is None.

        Returns
        -------
        None.

        """
        dataset = get_standard_dataset('ellipses', impl=self.impl)
        self.ray_trafo = dataset.get_ray_trafo(impl=self.impl)
        
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.ellipses_train = dataset.create_torch_dataset(part='train',
                                    reshape=((1,) + dataset.space[0].shape,
                                    (1,) + dataset.space[1].shape))
            
            self.ellipses_val = dataset.create_torch_dataset(part='validation',
                                    reshape=((1,) + dataset.space[0].shape,
                                        (1,) + dataset.space[1].shape))

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.ellipses_test = dataset.create_torch_dataset(part='test',
                                    reshape=((1,) + dataset.space[0].shape,
                                        (1,) + dataset.space[1].shape))

    def train_dataloader(self):
        """
        Data loader for the training data.

        Returns
        -------
        DataLoader
            Training data loader.

        """
        return DataLoader(self.ellipses_train, batch_size=self.batch_size,
                          num_workers=self.num_data_loader_workers,
                          pin_memory=True)

    def val_dataloader(self):
        """
        Data loader for the validation data.

        Returns
        -------
        DataLoader
            Validation data loader.

        """
        return DataLoader(self.ellipses_val, batch_size=self.batch_size,
                          num_workers=self.num_data_loader_workers,
                          pin_memory=True)

    def test_dataloader(self):
        """
        Data loader for the test data.

        Returns
        -------
        DataLoader
            Test data loader.

        """
        return DataLoader(self.ellipses_test, batch_size=self.batch_size,
                          num_workers=self.num_data_loader_workers,
                          pin_memory=True)
