import pytorch_lightning as pl 
from torch.utils.data import DataLoader
from dival import get_standard_dataset


class LoDoPaBDataModule(pl.LightningDataModule):

    def __init__(self, batch_size:int = 4, impl:str = 'astra_cuda',
                 sorted_by_patient:bool = True,
                 num_data_loader_workers:int = 8):
        """
        Initialize the data module for the LoDoPaB-CT dataset.

        Parameters
        ----------
        batch_size : int, optional
            Size of a mini batch.
            The default is 4.
        impl : str, optional
            Which ASTRA implementation should be used.
            The default is 'astra_cuda'.
        sorted_by_patient : bool, optional
            Sort data by patient. Important if only a subset of the data is
            used.
            The default is True.
        num_data_loader_workers : int, optional
            Number of workers.
            The default is 8.

        Returns
        -------
        None.

        """
        super().__init__()

        self.batch_size = batch_size
        self.impl = impl
        self.sorted_by_patient = sorted_by_patient
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
        
        dataset = get_standard_dataset('lodopab', impl=self.impl,
                                sorted_by_patient=self.sorted_by_patient)
        self.ray_trafo = dataset.get_ray_trafo(impl=self.impl)
        
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.lodopab_train = dataset.create_torch_dataset(part='train',
                                    reshape=((1,) + dataset.space[0].shape,
                                    (1,) + dataset.space[1].shape))
            
            self.lodopab_val = dataset.create_torch_dataset(part='validation',
                                    reshape=((1,) + dataset.space[0].shape,
                                        (1,) + dataset.space[1].shape))

            self.dims = tuple(self.lodopab_train[0][0].shape)

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.lodopab_test = dataset.create_torch_dataset(part='test',
                                    reshape=((1,) + dataset.space[0].shape,
                                        (1,) + dataset.space[1].shape))

            self.dims = tuple(self.lodopab_test[0][0].shape)

    def train_dataloader(self):
        """
        Data loader for the training data.

        Returns
        -------
        DataLoader
            Training data loader.

        """
        return DataLoader(self.lodopab_train, batch_size=self.batch_size,
                          num_workers=self.num_data_loader_workers,
                          shuffle=True, pin_memory=True)

    def val_dataloader(self):
        """
        Data loader for the validation data.

        Returns
        -------
        DataLoader
            Validation data loader.

        """
        return DataLoader(self.lodopab_val, batch_size=self.batch_size,
                          num_workers=self.num_data_loader_workers,
                          shuffle=False, pin_memory=True)

    def test_dataloader(self):
        """
        Data loader for the test data.

        Returns
        -------
        DataLoader
            Test data loader.

        """
        return DataLoader(self.lodopab_test, batch_size=self.batch_size,
                          num_workers=self.num_data_loader_workers,
                          shuffle=False, pin_memory=True)
