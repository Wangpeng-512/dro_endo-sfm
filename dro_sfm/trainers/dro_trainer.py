
import os
import torch
from dro_sfm.trainers.base_trainer import BaseTrainer, sample_to_cuda
from dro_sfm.utils.config import prep_logger_and_checkpoint
from dro_sfm.utils.logging import print_config
from dro_sfm.utils.logging import AvgMeter
from tqdm import tqdm
from dro_sfm.utils.logging import prepare_dataset_prefix
from dro_sfm.utils.depth import viz_inv_depth
from torch.utils.tensorboard import SummaryWriter

class DROTrainer():
    def __init__(self, min_epochs=0, max_epochs=50,
                 checkpoint=None,**kwargs):

        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 1)))
        # torch.cuda.set_device('cuda')
        torch.backends.cudnn.benchmark = True
        self.min_epochs = min_epochs
        self.max_epochs = max_epochs

        self.checkpoint = checkpoint
        self.module = None
        self.avg_loss = AvgMeter(50)
        self.avg_loss2 = AvgMeter(50)
        self.dtype = kwargs.get("dtype", None)  # just for test for now
        self.logger = SummaryWriter(log_dir='./log/events', flush_secs=30)


    def fit(self, module):

        # Prepare module for training
        module.trainer = self
        # Update and print module configuration
        prep_logger_and_checkpoint(module)
        print_config(module.config)

        # Send module to GPU
        module = module.to('cuda')
        # Configure optimizer and scheduler
        module.configure_optimizers()


        optimizer = module.optimizer
        scheduler = module.scheduler

        # Get train and val dataloaders
        train_dataloader = module.train_dataloader()
        val_dataloaders = module.val_dataloader()

        # Epoch loop
        for epoch in range(module.current_epoch, self.max_epochs):
            # Train
            self.train(train_dataloader, module, optimizer)
            # Validation
            validation_output = self.validate(val_dataloaders, module)
            # Check and save model
            self.check_and_save(module, validation_output)
            # Update current epoch
            module.current_epoch += 1
            # Take a scheduler step
            scheduler.step()

    def train(self, dataloader, module, optimizer):
        # Set module to train
        module.train()
        # Shuffle dataloader sampler
        if hasattr(dataloader.sampler, "set_epoch"):
            dataloader.sampler.set_epoch(module.current_epoch)
        # Prepare progress bar
        progress_bar = self.train_progress_bar(
            dataloader, module.config.datasets.train)
        # Start training loop
        # outputs = []
        # For all batches
        for i, batch in progress_bar:
            # Reset optimizer
            optimizer.zero_grad()
            # Send samples to GPU and take a training step
            batch = sample_to_cuda(batch)
            output = module.training_step(batch, i)

            #**********************************************************************
            if i%100 == 0:
                self.logger.add_image("image", batch['rgb'][0], global_step = i + module.current_epoch * len(dataloader), dataformats='CHW')
                viz_depth = []
                import numpy as np
                for inv_depth in output['inv_depth']:
                    depth = np.uint8(viz_inv_depth(inv_depth[0]) * 255)
                    depth = np.expand_dims(depth, 0)
                    viz_depth.append(depth)
                viz_depth =  np.concatenate(viz_depth, axis=0) 
                self.logger.add_images("iters", viz_depth, global_step = i + module.current_epoch * len(dataloader), dataformats='NHWC')

            if output is None:
                print("skip this training step.....")
                continue
            # Backprop through loss and take an optimizer step
            output['loss'].backward()
            optimizer.step()
            # Append output to list of outputs
            output['loss'] = output['loss'].detach()
            # outputs.append(output)
            # Update progress bar if in rank 0
            
            progress_bar.set_description(
                'Epoch {} | Avg.Loss {:.4f} Loss2 {:.4f}'.format(
                    module.current_epoch, self.avg_loss(output['loss'].item()),
                    self.avg_loss2(output['metrics']['pose_loss'].item() if 'pose_loss' in output['metrics'] else 0.0)))
        # Return outputs for epoch end
        # return module.training_epoch_end(outputs)

    def validate(self, dataloaders, module):
        # Set module to eval
        module.eval()
        # Start validation loop
        all_outputs = []
        # For all validation datasets
        for n, dataloader in enumerate(dataloaders):
            # Prepare progress bar for that dataset
            progress_bar = self.val_progress_bar(
                dataloader, module.config.datasets.validation, n)
            outputs = []
            # For all batches
            for i, batch in progress_bar:
                # Send batch to GPU and take a validation step
                batch = sample_to_cuda(batch)
                output = module.validation_step(batch, i, n)
                # Append output to list of outputs
                outputs.append(output)
            # Append dataset outputs to list of all outputs
            all_outputs.append(outputs)
        # Return all outputs for epoch end
        return module.validation_epoch_end(all_outputs)

    def test(self, module):
        # Send module to GPU
        module = module.to('cuda', dtype=self.dtype)
        # Get test dataloaders
        test_dataloaders = module.test_dataloader()
        # Run evaluation
        self.evaluate(test_dataloaders, module)

    @torch.no_grad()
    def evaluate(self, dataloaders, module):
        # Set module to eval
        module.eval()
        # Start evaluation loop
        all_outputs = []
        # For all test datasets
        for n, dataloader in enumerate(dataloaders):
            # Prepare progress bar for that dataset
            progress_bar = self.val_progress_bar(
                dataloader, module.config.datasets.test, n)
            outputs = []
            # For all batches
            for i, batch in progress_bar:
                # Send batch to GPU and take a test step
                batch = sample_to_cuda(batch, self.dtype)
                output = module.test_step(batch, i, n)
                # Append output to list of outputs
                outputs.append(output)
            # Append dataset outputs to list of all outputs
            all_outputs.append(outputs)
        # Return all outputs for epoch end
        return module.test_epoch_end(all_outputs)

    def check_and_save(self, module, output):
        if self.checkpoint:
            self.checkpoint.check_and_save(module, output)

    def train_progress_bar(self, dataloader, config, ncols=120):
        return tqdm(enumerate(dataloader, 0),
                    unit=' images', unit_scale=config.batch_size,
                    total=len(dataloader), smoothing=0,
                    ncols=ncols
                    )

    def val_progress_bar(self, dataloader, config, n=0, ncols=120):
        return tqdm(enumerate(dataloader, 0),
                    unit=' images', unit_scale= config.batch_size,
                    total=len(dataloader), smoothing=0,
                    ncols=ncols,
                    desc=prepare_dataset_prefix(config, n)
                    )

    def test_progress_bar(self, dataloader, config, n=0, ncols=120):
        return tqdm(enumerate(dataloader, 0),
                    unit=' images', unit_scale=config.batch_size,
                    total=len(dataloader), smoothing=0,
                    ncols=ncols,
                    desc=prepare_dataset_prefix(config, n)
                    )
