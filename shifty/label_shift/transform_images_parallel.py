# Transform a dataset of images in parallel using python's multiprocessing library
# Based on code by Gerardo Duran-Martin at 
# https://github.com/probml/shift-happens/blob/main/shift_happens/gendist/gendist/processing.py
# https://github.com/probml/shift-happens/blob/main/shift_happens/gendist/experiments/multiprocess_test.py

import jax
import numpy as np
import jax.numpy as jnp
from multiprocessing import Pool

import torchvision
import numpy as np
from augly import image

# DataAugmentationFactory
class Factory:
    """
    This is a base library to process / transform the elements of a numpy
    array according to a given function. To be used with gendist.TrainingConfig
    """
    def __init__(self, processor):
        self.processor = processor
    
    def __call__(self, img, configs, n_processes=90):
        return self.process_multiple_multiprocessing(img, configs, n_processes)

    def process_single(self, X, *args, **kwargs):
        """
        Process a single element.
        Paramters
        ---------
        X: np.array
            A single numpy array
        kwargs: dict/params
            Processor's configuration parameters
        """
        return self.processor(X, *args, **kwargs)

    def process_multiple(self, X_batch, configurations):
        """
        Process all elements of a numpy array according to a list
        of configurations.
        Each image is processed according to a configuration.
        """
        X_out = []
        n_elements = len(X_batch)
                    
        for X, configuration in zip(X_batch, configurations):
            X_processed = self.process_single(X, **configuration)
            X_out.append(X_processed)
            
        X_out = np.stack(X_out, axis=0)
        return X_out
    
    def process_multiple_multiprocessing(self, X_dataset, configurations, n_processes):
        """
        Process elements in a numpy array in parallel.
        Parameters
        ----------
        X_dataset: array(N, ...)
            N elements of arbitrary shape
        configurations: list
            List of configurations to apply to each element. Each
            element is a dict to pass to the processor.
        n_processes: int
            Number of cores to use
        """
        num_elements = len(X_dataset)
        if type(configurations) == dict:
            configurations = [configurations] * num_elements

        dataset_proc = np.array_split(X_dataset, n_processes)
        config_split = np.array_split(configurations, n_processes)
        elements = zip(dataset_proc, config_split)

        with Pool(processes=n_processes) as pool:    
            dataset_proc = pool.starmap(self.process_multiple, elements)
            dataset_proc = np.concatenate(dataset_proc, axis=0)
        pool.join()

        return dataset_proc.reshape(num_elements, -1)

def processor(X, angle):
    X_shift = image.aug_np_wrapper(X, image.rotate, degrees=angle)
    size_im = X_shift.shape[0]
    size_pad = (28 - size_im) // 2
    size_pad_mod = (28 - size_im) % 2
    X_shift = np.pad(X_shift, (size_pad, size_pad + size_pad_mod))
    return X_shift


def main():
    from time import time

    init_time = time()
    mnist_train = torchvision.datasets.MNIST(root="~/data", train=True, download=True)
    images = np.array(mnist_train.data) / 255.0
    images = images[:200, :, :]

    n_configs = 6
    degrees = np.linspace(0, 360, n_configs)
    configs = [{"angle": float(angle)} for angle in degrees]
    process =  Factory(processor)
    images_proc = process(images, configs, n_processes=90)
    end_time = time()
    
    print(f"Time elapsed: {end_time - init_time:.2f}s")
    print(images_proc.shape)


if __name__ == "__main__":
    main()