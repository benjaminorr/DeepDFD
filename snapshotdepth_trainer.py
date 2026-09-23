import os
from argparse import ArgumentParser

import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import ConcatDataset, DataLoader

from datasets.dualpixel import DualPixel
from datasets.sceneflow import SceneFlow
from snapshotdepth import SnapshotDepth
from util.log_manager import LogManager, find_resume_checkpoint

seed_everything(123)


def prepare_data(hparams):
    image_sz = hparams.image_sz
    crop_width = hparams.crop_width
    augment = hparams.augment
    randcrop = hparams.randcrop

    padding = 0
    val_idx = 3994
    sf_train_dataset = SceneFlow('train',
                                 (image_sz + 4 * crop_width,
                                  image_sz + 4 * crop_width),
                                 is_training=True,
                                 randcrop=randcrop, augment=augment, padding=padding,
                                 singleplane=False)
    sf_train_dataset = torch.utils.data.Subset(sf_train_dataset,
                                               range(val_idx, len(sf_train_dataset)))

    sf_val_dataset = SceneFlow('train',
                               (image_sz + 4 * crop_width,
                                image_sz + 4 * crop_width),
                               is_training=False,
                               randcrop=randcrop, augment=augment, padding=padding,
                               singleplane=False)
    sf_val_dataset = torch.utils.data.Subset(sf_val_dataset, range(val_idx))

    if hparams.mix_dualpixel_dataset:
        dp_train_dataset = DualPixel('train',
                                     (image_sz + 4 * crop_width,
                                      image_sz + 4 * crop_width),
                                     is_training=True,
                                     randcrop=randcrop, augment=augment, padding=padding)
        dp_val_dataset = DualPixel('val',
                                   (image_sz + 4 * crop_width,
                                    image_sz + 4 * crop_width),
                                   is_training=False,
                                   randcrop=randcrop, augment=augment, padding=padding)

        train_dataset = ConcatDataset([dp_train_dataset, sf_train_dataset])
        val_dataset = ConcatDataset([dp_val_dataset, sf_val_dataset])

        n_sf = len(sf_train_dataset)
        n_dp = len(dp_train_dataset)
        sample_weights = torch.cat([1. / n_dp * torch.ones(n_dp, dtype=torch.double),
                                    1. / n_sf * torch.ones(n_sf, dtype=torch.double)], dim=0)
        sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))

        train_dataloader = DataLoader(train_dataset, batch_size=hparams.batch_sz, sampler=sampler,
                                      num_workers=hparams.num_workers, shuffle=False, pin_memory=True)
        val_dataloader = DataLoader(val_dataset, batch_size=hparams.batch_sz,
                                    num_workers=hparams.num_workers, shuffle=False, pin_memory=True)
    else:
        train_dataset = sf_train_dataset
        val_dataset = sf_val_dataset
        train_dataloader = DataLoader(train_dataset, batch_size=hparams.batch_sz,
                                      num_workers=hparams.num_workers, shuffle=True, pin_memory=True)
        val_dataloader = DataLoader(val_dataset, batch_size=hparams.batch_sz,
                                    num_workers=hparams.num_workers, shuffle=False, pin_memory=True)

    return train_dataloader, val_dataloader


def _str_to_bool(s):
    if s.lower() in ('true', 't', 'yes', 'y', '1'):
        return True
    if s.lower() in ('false', 'f', 'no', 'n', '0'):
        return False
    raise ValueError(f'Cannot interpret {s} as a boolean.')


def _int_or_float(s):
    return float(s) if '.' in s else int(s)


def _parse_devices_arg(devices):
    """'auto' -> 'auto', '4' -> 4, '0,1' -> [0, 1]"""
    if devices == 'auto':
        return devices
    if ',' in devices:
        return [int(d) for d in devices.split(',') if d]
    return int(devices)


def trainer_kwargs_from_args(args):
    # Lightning dropped Trainer.from_argparse_args, and the old --gpus/--distributed_backend/--replace_sampler_ddp
    # flags. Translate them here so that the original command lines keep working.
    accelerator = args.accelerator
    devices = _parse_devices_arg(args.devices)
    if args.gpus is not None:
        gpus = _parse_devices_arg(args.gpus)
        if gpus == 0:
            accelerator, devices = 'cpu', 1
        else:
            accelerator, devices = 'gpu', gpus
    strategy = args.distributed_backend if args.distributed_backend is not None else args.strategy
    if accelerator != 'gpu':
        # A distributed strategy (e.g. ddp) and sync_batchnorm only make sense across multiple GPUs; on CPU
        # (or a single device) force a plain single-process run instead of failing inside DDP/SyncBatchNorm setup.
        strategy = 'auto'

    kwargs = dict(
        default_root_dir=args.default_root_dir,
        max_epochs=args.max_epochs,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        precision=args.precision,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        num_sanity_val_steps=args.num_sanity_val_steps,
        log_every_n_steps=args.log_every_n_steps,
    )
    if args.max_steps is not None:
        kwargs['max_steps'] = args.max_steps
    if args.replace_sampler_ddp is not None:
        kwargs['use_distributed_sampler'] = args.replace_sampler_ddp
    return kwargs


def main(args):
    # A fixed version keeps restarts in the same log directory, so that the checkpoint can be found for resuming.
    logger = TensorBoardLogger(args.default_root_dir,
                               name=args.experiment_name,
                               version=0)

    logmanager_callback = LogManager()

    ckpt_dir = os.path.join(logger.log_dir, 'checkpoints')
    checkpoint_callback = ModelCheckpoint(
        verbose=True,
        monitor='val_loss',
        dirpath=ckpt_dir,
        filename='{epoch}-{val_loss:.4f}',
        save_top_k=1,
        save_last=True,
        every_n_epochs=1,
        mode='min',
    )

    ckpt_path = find_resume_checkpoint(ckpt_dir)
    if ckpt_path is not None:
        print(f'Resuming from {ckpt_path} (use a different --experiment_name to start a new experiment)')
    else:
        print(f'Starting a new experiment and logging at \n {os.path.expanduser(logger.log_dir)}')

    model = SnapshotDepth(hparams=args, log_dir=logger.log_dir)
    train_dataloader, val_dataloader = prepare_data(hparams=args)

    trainer_kwargs = trainer_kwargs_from_args(args)
    # SyncBatchNorm only works across GPU modules; trainer_kwargs_from_args() already forces a single-process
    # strategy off of GPU, so only request it when we're actually on the gpu accelerator.
    sync_batchnorm = trainer_kwargs['accelerator'] == 'gpu'

    trainer = Trainer(
        logger=logger,
        callbacks=[logmanager_callback, checkpoint_callback],
        sync_batchnorm=sync_batchnorm,
        benchmark=True,
        **trainer_kwargs,
    )
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, ckpt_path=ckpt_path)


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)

    parser.add_argument('--experiment_name', type=str, default='LearnedDepth')
    parser.add_argument('--mix_dualpixel_dataset', dest='mix_dualpixel_dataset', action='store_true')
    parser.add_argument('--no-mix_dualpixel_dataset', dest='mix_dualpixel_dataset', action='store_false')
    parser.set_defaults(mix_dualpixel_dataset=True)

    # Trainer parameters
    parser.add_argument('--default_root_dir', type=str, default='data/logs')
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--max_steps', type=int, default=None)
    parser.add_argument('--accelerator', type=str, default='auto')
    parser.add_argument('--devices', type=str, default='1')
    parser.add_argument('--strategy', type=str, default='auto')
    parser.add_argument('--precision', type=str, default='32-true')
    parser.add_argument('--limit_train_batches', type=_int_or_float, default=1.0)
    parser.add_argument('--limit_val_batches', type=_int_or_float, default=1.0)
    parser.add_argument('--num_sanity_val_steps', type=int, default=2)
    parser.add_argument('--log_every_n_steps', type=int, default=50)
    # Old-style flags from pytorch-lightning 1.0 (used in the README and the paper)
    parser.add_argument('--gpus', type=str, default=None, help='Deprecated. Use --accelerator and --devices.')
    parser.add_argument('--distributed_backend', type=str, default=None, help='Deprecated. Use --strategy.')
    parser.add_argument('--replace_sampler_ddp', type=_str_to_bool, nargs='?', const=True, default=None)

    parser = SnapshotDepth.add_model_specific_args(parser)

    args = parser.parse_args()

    main(args)
