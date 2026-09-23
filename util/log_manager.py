import os

from lightning.pytorch.callbacks import Callback


def find_resume_checkpoint(ckpt_dir):
    """Return the checkpoint to resume from (last.ckpt, else the newest *.ckpt), or None.

    This has to run before Trainer.fit(), because the ckpt_path of a fit() call cannot be changed from a callback.
    """
    if not os.path.isdir(ckpt_dir):
        return None
    last_ckpt = os.path.join(ckpt_dir, 'last.ckpt')
    if os.path.exists(last_ckpt):
        return last_ckpt
    ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith('.ckpt')]
    if not ckpt_files:
        return None
    ckpt_files.sort(key=lambda f: os.path.getmtime(os.path.join(ckpt_dir, f)), reverse=True)
    return os.path.join(ckpt_dir, ckpt_files[0])


class LogManager(Callback):

    def setup(self, trainer, pl_module, stage):
        if trainer.is_global_zero:
            print("*" * 30)
            print('log_dir', os.path.expanduser(trainer.logger.log_dir))
            print("*" * 30)

    def on_exception(self, trainer, pl_module, exception):
        if isinstance(exception, KeyboardInterrupt):
            # Every rank must call save_checkpoint (it synchronizes); only rank 0 writes the file.
            ckpt_path = os.path.join(trainer.logger.log_dir, 'checkpoints', 'interrupted_model.ckpt')
            trainer.save_checkpoint(ckpt_path)
            if trainer.is_global_zero:
                print(f'Saved a checkpoint to {ckpt_path}')
