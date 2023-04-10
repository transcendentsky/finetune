import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP
from .trainer import SamLearner, DataEngine, sam_model_registry
from torch.utils.data import DataLoader, Dataset
from tutils.trainer.trainer_abstract import AbstractTrainer
from tutils import tfilename
from tutils.tutils.ttimer import tenum, timer
from tutils.trainer.recorder import Recorder
from tutils.tutils import tfilename, CSVLogger, MultiLogger
from monai.data import CacheDataset, DataLoader, Dataset
from rich.progress import track
from torch.cuda.amp import autocast, GradScaler

def get_logger(config):
    config_base = config['base']
    config_logger = config['logger']
    logger = MultiLogger(logdir=config_base['runs_dir'], 
                        record_mode=config_logger.get('record_mode', None), 
                        tag=config_base['tag'], 
                        extag=config_base.get('experiment', None),
                        action=config_logger.get('action', 'k')) # backup config.yaml
    return logger

class DDPTrainer(AbstractTrainer):
    def __init__(self, config, tester=None, monitor=None, rank='cuda', world_size=0, logger=None):
        super().__init__(config, tester, monitor, rank, world_size)
        self.logger = logger
        if self.logging_available:
            print("Logger at Process(rank=0)")
            self.recorder = Recorder(reduction=self.recorder_mode)
            self.recorder_test = Recorder(reduction=self.recorder_mode)
            self.logger = None
            self.csvlogger = CSVLogger(tfilename(self.runs_dir, "best_record"))
            self.csvlogger_all = CSVLogger(tfilename(self.runs_dir, "all_record"))
            self.monitor = monitor
            self.tester = tester
            
            self.logger = get_logger(config)
            assert self.logger is not None, f"Got rank {self.rank}"
        
        if self.use_amp:
            self.scalar = GradScaler()
            print("Debug settings: use amp=",self.use_amp)

    def init_model(self, model, trainset, **kwargs):
        assert len(trainset) > 0 , f"Got {len(trainset)}"

        # Use CacheDataset
        # trainset = CacheDataset(trainset, num_workers=12, cache_rate=0.5)
        self.trainloader = DataLoader(dataset=trainset,
                                      batch_size=self.batch_size,
                                      num_workers=8,
                                      shuffle=True,
                                      drop_last=True,
                                      pin_memory=True)
        if self.load_pretrain_model:
            model.module.load()
        rank = self.rank
        model = model.to(rank)
        ddp_model = DDP(model, device_ids=[rank])
        return ddp_model

    def configure_optim(self, model, **kwargs):
        # Set optimizer and scheduler
        optim_configs = model.module.configure_optimizers()
        assert isinstance(optim_configs, dict)
        optimizer = optim_configs['optimizer']
        scheduler = optim_configs['scheduler']

        if self.load_optimizer:
            start_epoch = model.module.load_optim(optimizer)
        else:
            start_epoch = self.start_epoch
        return optimizer, scheduler, start_epoch

    def save(self, model, epoch, type=None, optimizer=None, **kwargs):
        if self.logging_available:
            if type is None:
                # if self.save_interval > 0 and epoch % self.save_interval == 0:
                save_name = "/ckpt/model_epoch_{}.pth".format(epoch)
                model.module.save(tfilename(self.runs_dir, save_name), epoch=epoch)
                self.logger.info(f"Epoch {epoch}: Save model to ``{save_name}``! ")
            elif type == 'best':
                # save_name = "/ckpt/best_model_epoch_{}.pth".format(epoch)
                save_name2 = "/ckpt_v/model_best.pth"
                # model.save(tfilename(self.runs_dir, save_name), epoch=epoch, is_best=True)
                model.module.save(tfilename(self.runs_dir, save_name2), epoch=epoch, is_best=True)
                self.logger.info(f"[Best model] Epoch {epoch}: Save model to ``{save_name2}``! ")
            elif type == 'latest':
                if self.save_interval > 0 and epoch % self.save_interval == 0:
                    save_name = "/ckpt_v/model_latest.pth"
                    model.module.save(tfilename(self.runs_dir, save_name), epoch=epoch, is_latest=True)
                    save_optim_name = "/ckpt/optim_latest.pth"
                    model.module.save_optim(tfilename(self.runs_dir, save_optim_name), optimizer=optimizer, epoch=epoch)
                    self.logger.info(f"Epoch {epoch}: Save checkpoint to ``{save_name}``")


    def train(self, model, trainloader, epoch, optimizer, scheduler=None, do_training_log=True):
        model.train()
        out = {}
        if do_training_log and self.logging_available:
            self.recorder.clear()
            time_record = 0.1111
            self.timer_batch()

        success_count = 0
        failed_count = 0
        for load_time, batch_idx, data in tenum(trainloader):
            optimizer.zero_grad()
            self.timer_data()
            # training steps
            for k, v in data.items():
                if type(v) == torch.Tensor:
                    data[k] = v.to(self.rank)
            time_data_cuda = self.timer_data()
            if self.use_amp:           
                with autocast():
                    self.timer_net()
                    out = model.module.training_step(data, batch_idx, epoch=epoch)
                    assert isinstance(out, dict)
                    time_fd = self.timer_net()
                    loss = out['loss']
                    self.scalar.scale(loss).backward()
                    self.scalar.step(optimizer)
                    self.scalar.update()
                    time_bp = self.timer_net()
            else:
                self.timer_net()
                out = model.module.training_step(data, batch_idx, epoch=epoch)
                if out['loss'] is None:
                    failed_count += 1
                    continue
                if torch.isnan(out['loss']):
                    print("Ignore Nan Value: ", out['loss'])
                    failed_count += 1
                    # raise ValueError(f"Get loss: {out['loss']}")
                assert isinstance(out, dict)
                time_fd = self.timer_net()
                loss = out['loss']
                loss.backward()
                optimizer.step()
                time_bp = self.timer_net()
                success_count += 1

            time_batch = self.timer_batch()
            # batch logger !
            if self.logging_available and do_training_log:
                out['time_load'] = load_time
                out['time_cuda'] = time_data_cuda
                out['time_forward'] = time_fd
                out['time_bp'] = time_bp
                out['time_record'] = time_record
                out['time_batch'] = time_batch
                self.timer_data()
                self.recorder.record(out)
                time_record = self.timer_data()
                print(f"Epoch: {epoch}. Processing batch_idx:{batch_idx} / {len(trainloader)}, time_load: {load_time}", end='\r')
                # for debug !
                if epoch == 0:
                    if self.logging_available:
                        self.logger.info("[*] Debug Checking Pipeline !!!")
                    return
        if scheduler is not None:
            scheduler.step()        
    

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


def ddp_train(rank, world_size, config, tconfig):    
    setup(rank, world_size)
    CACHE_DISK_DIR="../cache/data2d_3/"
    sam_checkpoint = "../segment-anything/sam_vit_h_4b8939.pth"
    device = "cuda"
    model_type = "default"
    data_engine = DataEngine(dirpath=CACHE_DISK_DIR, img_size=(1024,1024))

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    learner = SamLearner(sam_model=sam, config=config, data_engine=data_engine)

    # create model and move it to GPU with id rank
    ddp_trainer = DDPTrainer(config=config, rank=rank, world_size=world_size)
    ddp_trainer.fit(learner, data_engine)

    cleanup()

def run_demo(demo_fn, world_size, config, tconfig):
    mp.spawn(demo_fn,
             args=(world_size,config,tconfig,),
             nprocs=world_size,
             join=True)
    

if __name__ == "__main__":
    # from tutils import TConfig
    from tutils.tutils.initializer import TConfig
    import argparse

    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./configs/config.yaml")
    parser.add_argument("--func", default="train")

    tconfig = TConfig(file=__file__, mode="sc", parser=parser)
    config = tconfig.get_config()
    # logger = tconfig.get_logger()
    # args = tconfig.get_args()

    # args = trans_args(parser)
    # logger, config = trans_init(args, file=__file__, no_logger=True)

    run_demo(ddp_train, world_size, config, None)