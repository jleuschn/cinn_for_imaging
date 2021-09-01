import pytorch_lightning as pl 
import odl
from torch.functional import norm 
from torch.utils.data import DataLoader

from cinn_for_imaging.datasets.fast_mri.mri_data import SliceDataset, to_tensor

import os 
import torch
import numpy as np 
from typing import Dict, Optional, Sequence, Tuple, Union

def normalize(data, mean, stddev, eps=0.):
    """
    Normalize the given tensor using:
        (data - mean) / (stddev + eps)
    Args:
        data (torch.Tensor): Input data to be normalized
        mean (float): Mean value
        stddev (float): Standard deviation
        eps (float): Added to stddev to prevent dividing by zero
    Returns:
        torch.Tensor: Normalized tensor
    """
    return (data - mean) / (stddev + eps)


def normalize_instance(data, eps=0.):
    """
        Normalize the given tensor using:
            (data - mean) / (stddev + eps)
        where mean and stddev are computed from the data itself.
        Args:
            data (torch.Tensor): Input data to be normalized
            eps (float): Added to stddev to prevent dividing by zero
        Returns:
            torch.Tensor: Normalized tensor
        """
    mean = data.mean()
    std = data.std()
    return normalize(data, mean, std, eps), mean, std


class FastMRIDataModule(pl.LightningDataModule):

    def __init__(self, batch_size:int = 4, num_data_loader_workers:int = 8):
        """
        Initialize the data module for the LoDoPaB-CT dataset.

        Parameters
        ----------
        batch_size : int, optional
            Size of a mini batch.
            The default is 4.
        num_data_loader_workers : int, optional
            Number of workers.
            The default is 8.

        Returns
        -------
        None.

        """
        super().__init__()

        super().__init__()
        
        self.batch_size = batch_size
        self.num_data_loader_workers = num_data_loader_workers
        self.data_range = [-6, 6]

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
        
        self.operator = odl.operator.default_ops.IdentityOperator(
                    odl.uniform_discr([0, 0], [1, 1], shape=(320, 320)))
        
        base_path = "/localdata/fast_mri"  # TODO adapt
        train_path = "singlecoil_train"
        val_path = "singlecoil_val"
        test_path = "singlecoil_test_v2"

        #mask_fun = RandomMaskFunc(center_fractions=[0.08, 0.04], accelerations=[4, 8]) # train both 4x and 8x acceleration 
        mask_fun = RandomMaskFunc(center_fractions=[0.08], accelerations=[4]) # train only on 4x acceleration

        train_transform = DataTransform(mask_func=mask_fun, use_seed=False)
        val_transform = DataTransform(mask_func=mask_fun)
        test_transform = DataTransform()

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.fastmri_train = SliceDataset(root=os.path.join(base_path, train_path), challenge="singlecoil", transform=train_transform)
            
            self.fastmri_val = SliceDataset(root=os.path.join(base_path, val_path), challenge="singlecoil", transform=val_transform)

            self.dims = tuple(self.fastmri_train[0][0].shape)

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.fastmri_test = SliceDataset(root=os.path.join(base_path, test_path), challenge="singlecoil", transform=test_transform)

            self.dims = tuple(self.fastmri_test[0][0].shape)

    def train_dataloader(self):
        """
        Data loader for the training data.

        Returns
        -------
        DataLoader
            Training data loader.

        """
        return DataLoader(self.fastmri_train, batch_size=self.batch_size,
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
        return DataLoader(self.fastmri_val, batch_size=self.batch_size,
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
        return DataLoader(self.fastmri_test, batch_size=self.batch_size,
                          num_workers=self.num_data_loader_workers,
                          shuffle=False, pin_memory=True)



class DataTransform:
    """
    Data Transformer for training cINN.
    """

    def __init__(
        self,
        mask_func  = None,
        use_seed: bool = True,
    ):
        """
        Args:
            mask_func: Optional; A function that can create a mask of
                appropriate shape.
            use_seed: If true, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """

        self.mask_func = mask_func
        self.use_seed = use_seed

    def __call__(
        self,
        kspace: np.ndarray,
        mask: np.ndarray,
        target: np.ndarray,
        attrs: Dict,
        fname: str,
        slice_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int, float]:
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data or (rows, cols) for single coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.
        Returns:
            tuple containing:
                image: Zero-filled input image.
                target: Target image converted to a torch.Tensor.
                mean: Mean value used for normalization.
                std: Standard deviation value used for normalization.
                fname: File name.
                slice_num: Serial number of the slice.
        """
        kspace = to_tensor(kspace)
        # check for max value
        max_value = attrs["max"] if "max" in attrs.keys() else 0.0
        acquisition = attrs["acquisition"]
        # apply mask
        if self.mask_func:
            seed = None if not self.use_seed else tuple(map(ord, fname))
            masked_kspace, mask = apply_mask(kspace, self.mask_func, seed)
        else:
            masked_kspace = kspace

        # inverse Fourier transform to get zero filled solution
        image = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(torch.view_as_complex(masked_kspace)), norm="ortho"))
        image = torch.fft.fftshift(image)

        # crop input to correct size
        if target is not None:
            crop_size = (target.shape[-2], target.shape[-1])
        else:
            crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])

        # check for FLAIR 203
        if image.shape[-2] < crop_size[1]:
            crop_size = (image.shape[-2], image.shape[-2])

        w_from = (image.shape[-2] - crop_size[0]) // 2
        h_from = (image.shape[-1] - crop_size[1]) // 2
        w_to = w_from + crop_size[0]
        h_to = h_from + crop_size[1]
        image = image[w_from:w_to, h_from:h_to]

        # normalize input
        image, mean, std = normalize_instance(image, eps=1e-11)
        image = image.clamp(-6, 6)

        # normalize target
        if target is not None:
            target = to_tensor(target)
            ## center crop
            w_from = (target.shape[-2] - crop_size[0]) // 2
            h_from = (target.shape[-1] - crop_size[1]) // 2
            w_to = w_from + crop_size[0]
            h_to = h_from + crop_size[1]
            target = target[w_from:w_to, h_from:h_to]
            #target, _, _ = normalize_instance(target, eps=1e-11)
            target = normalize(target, mean, std, eps=1e-11)
            target = target.clamp(-6, 6)
        else:
            target = torch.Tensor([0])

        return image.unsqueeze(0), target.unsqueeze(0), mean, std, fname, slice_num, acquisition

def apply_mask(data,mask_func,seed= None,padding = None):
    """
    Subsample given k-space by multiplying with a mask.
    Args:
        data: The input k-space data. This should have at least 3 dimensions,
            where dimensions -3 and -2 are the spatial dimensions, and the
            final dimension has size 2 (for complex values).
        mask_func: A function that takes a shape (tuple of ints) and a random
            number seed and returns a mask.
        seed: Seed for the random number generator.
        padding: Padding value to apply for mask.
    Returns:
        tuple containing:
            masked data: Subsampled k-space data
            mask: The generated mask
    """
    shape = np.array(data.shape)
    shape[:-3] = 1
    mask = mask_func(shape, seed)
    if padding is not None:
        mask[:, :, : padding[0]] = 0
        mask[:, :, padding[1] :] = 0  # padding value inclusive on right of zeros

    masked_data = data * mask + 0.0  # the + 0.0 removes the sign of the zeros

    return masked_data, mask


class RandomMaskFunc():
    """
    RandomMaskFunc creates a sub-sampling mask of a given shape.
    The mask selects a subset of columns from the input k-space data. If the
    k-space data has N columns, the mask picks out:
        1. N_low_freqs = (N * center_fraction) columns in the center
           corresponding to low-frequencies.
        2. The other columns are selected uniformly at random with a
        probability equal to: prob = (N / acceleration - N_low_freqs) /
        (N - N_low_freqs). This ensures that the expected number of columns
        selected is equal to (N / acceleration).
    It is possible to use multiple center_fractions and accelerations, in which
    case one possible (center_fraction, acceleration) is chosen uniformly at
    random each time the RandomMaskFunc object is called.
    For example, if accelerations = [4, 8] and center_fractions = [0.08, 0.04],
    then there is a 50% probability that 4-fold acceleration with 8% center
    fraction is selected and a 50% probability that 8-fold acceleration with 4%
    center fraction is selected.
    """
    def __init__(self, center_fractions, accelerations):
        """
        Args:
            center_fractions: Fraction of low-frequency columns to be retained.
                If multiple values are provided, then one of these numbers is
                chosen uniformly each time.
            accelerations: Amount of under-sampling. This should have the same
                length as center_fractions. If multiple values are provided,
                then one of these is chosen uniformly each time.
        """
        if not len(center_fractions) == len(accelerations):
            raise ValueError(
                "Number of center fractions should match number of accelerations"
            )

        self.center_fractions = center_fractions
        self.accelerations = accelerations
        self.rng = np.random.RandomState()  # pylint: disable=no-member

    def __call__(self, shape, seed=None):
        """
        Create the mask.
        Args:
            shape: The shape of the mask to be created. The shape should have
                at least 3 dimensions. Samples are drawn along the second last
                dimension.
            seed: Seed for the random number generator. Setting the seed
                ensures the same mask is generated each time for the same
                shape. The random state is reset afterwards.
        Returns:
            A mask of the specified shape.
        """
        if len(shape) < 3:
            raise ValueError("Shape should have 3 or more dimensions")

        num_cols = shape[-2]
        center_fraction, acceleration = self.choose_acceleration()

        # create the mask
        num_low_freqs = int(round(num_cols * center_fraction))
        prob = (num_cols / acceleration - num_low_freqs) / (
            num_cols - num_low_freqs
        )
        mask = self.rng.uniform(size=num_cols) < prob
        pad = (num_cols - num_low_freqs + 1) // 2
        mask[pad : pad + num_low_freqs] = True

        # reshape the mask
        mask_shape = [1 for _ in shape]
        mask_shape[-2] = num_cols
        mask = torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))

        return mask

    def choose_acceleration(self):
        """Choose acceleration based on class parameters."""
        choice = self.rng.randint(0, len(self.accelerations))
        center_fraction = self.center_fractions[choice]
        acceleration = self.accelerations[choice]

        return center_fraction, acceleration



if __name__ == "__main__":
    dataset = FastMRIDataModule(num_data_loader_workers=0, batch_size=1)
    dataset.prepare_data()
    dataset.setup()
    print(len(dataset.train_dataloader()))
    print(len(dataset.val_dataloader()))
    print(len(dataset.test_dataloader()))
    import matplotlib.pyplot as plt 
    for sample in dataset.train_dataloader():
        image = sample[0]
        gt = sample[1] 
        print(image.shape, gt.shape)

        fig, axes = plt.subplots(2,2)
        fig.suptitle("Training")
        im = axes[0,0].imshow(image[0,0,:,:], cmap="gray")
        fig.colorbar(im, ax=axes[0,0])
        axes[0,1].hist(image[0,0,:,:].numpy().ravel(), bins="auto")
        axes[0,0].set_title("zero filled ifft")
        axes[0,1].set_title("z.f. ifft histogram")


        im = axes[1,0].imshow(gt[0,0,:,:], cmap="gray")
        fig.colorbar(im, ax=axes[1,0])

        axes[1,1].hist(gt[0,0,:,:].numpy().ravel(), bins="auto")
        axes[1,0].set_title("truth")
        axes[1,1].set_title("truth histogram")

        plt.show()

        #image, mean, std = sample
        #print(image.shape, mean.shape, std.shape)

        #plt.figure()
        #plt.imshow(image.numpy()[0][0], cmap="gray")
        #plt.show()

        break


    for sample in dataset.val_dataloader():
        image = sample[0]
        gt = sample[1] 
        print(image.shape, gt.shape)
        fig, axes = plt.subplots(2,2)
        fig.suptitle("Validation")
        im = axes[0,0].imshow(image[0,0,:,:], cmap="gray")
        fig.colorbar(im, ax=axes[0,0])
        axes[0,1].hist(image[0,0,:,:].numpy().ravel(), bins="auto")
        axes[0,0].set_title("zero filled ifft")
        axes[0,1].set_title("z.f. ifft histogram")


        im = axes[1,0].imshow(gt[0,0,:,:], cmap="gray")
        fig.colorbar(im, ax=axes[1,0])

        axes[1,1].hist(gt[0,0,:,:].numpy().ravel(), bins="auto")
        axes[1,0].set_title("truth")
        axes[1,1].set_title("truth histogram")

        plt.show()

        #image, mean, std = sample
        #print(image.shape, mean.shape, std.shape)

        #plt.figure()
        #plt.imshow(image.numpy()[0][0], cmap="gray")
        #plt.show()

        #break

        """
        ## gradient descent 
        xk = torch.zeros(*masked_kspace.shape)
        for i in range(15):
            xk = xk - torch.fft.ifft2(mask*(torch.fft.fft2(xk)-masked_kspace))
        xk = torch.fft.ifftshift(xk)
        xk = torch.abs(xk)

        crop_size = (target.shape[-2], target.shape[-1])

        w_from = (xk.shape[-2] - crop_size[0]) // 2
        h_from = (xk.shape[-1] - crop_size[1]) // 2
        w_to = w_from + crop_size[0]
        h_to = h_from + crop_size[1]
        xk = xk[:,:,w_from:w_to, h_from:h_to]

        xk, mean, std = normalize_instance(xk, eps=1e-11)
        xk = xk.clamp(-6, 6)
        print(xk.shape)

        import matplotlib.pyplot as plt 
        fig, (ax1, ax2, ax3) = plt.subplots(1,3)
        im = ax1.imshow(image[0,0,:,:], cmap="gray")
        ax1.set_title("zero filled ifft")
        fig.colorbar(im, ax=ax1)
        im = ax2.imshow(target[0,0, :,:], cmap="gray")
        ax2.set_title("ground truth data")
        fig.colorbar(im, ax=ax2)
        im = ax3.imshow(xk[0,0,:,:], cmap="gray")
        fig.colorbar(im, ax=ax3)
        ax3.set_title("gradient descent")
        plt.show()
        """


    """
    for sample in dataset.test_dataloader():
        image, mask, kspace = sample
        print(image.shape, mask.shape, kspace.shape)
        import matplotlib.pyplot as plt 
        fig, (ax1, ax2, ax3) = plt.subplots(1,3)
        im = ax1.imshow(image[0,0,:,:], cmap="gray")
        fig.colorbar(im, ax=ax1)
        im = ax2.imshow(mask[0,0, :,:], cmap="gray", aspect="auto", interpolation=None)
        fig.colorbar(im, ax=ax2)

        im = ax3.imshow(torch.log(torch.abs(kspace[0,0, :,:])), cmap="gray", aspect="auto", interpolation=None)
        fig.colorbar(im, ax=ax3)
        plt.show()
    """