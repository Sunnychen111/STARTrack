import os
from collections import OrderedDict
from lib.train.trainers import BaseTrainer
from lib.train.admin import AverageMeter, StatValue
from lib.train.admin import TensorboardWriter
import torch
import time
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
import lib.utils.misc as misc
from tqdm import tqdm

class LTRTrainer(BaseTrainer):
    def __init__(self, actor, loaders, optimizer, settings, lr_scheduler=None, use_amp=False):
        """
        args:
            actor - The actor for training the network
            loaders - list of dataset loaders, e.g. [train_loader, val_loader].
            optimizer - The optimizer used for training, e.g. Adam
            settings - Training settings
            lr_scheduler - Learning rate scheduler
        """
        super().__init__(actor, loaders, optimizer, settings, lr_scheduler)
        self._set_default_settings()

        # Initialize statistics variables
        self.stats = OrderedDict({loader.name: None for loader in self.loaders})

        # Initialize tensorboard
        if settings.local_rank in [-1, 0]:
            tensorboard_writer_dir = os.path.join(self.settings.env.tensorboard_dir, self.settings.project_path)
            if not os.path.exists(tensorboard_writer_dir):
                os.makedirs(tensorboard_writer_dir)
            self.tensorboard_writer = TensorboardWriter(tensorboard_writer_dir, [l.name for l in loaders])

        self.move_data_to_gpu = getattr(settings, 'move_data_to_gpu', True)
        self.settings = settings
        self.use_amp = use_amp
        if use_amp:
            self.scaler = GradScaler()

        # [修复] 健壮的系统检查 (防止 save_dir 报错)
        if misc.is_main_process():
            print(f"\n{'='*20} System Check {'='*20}")
            print(f"Log File:   {self.settings.log_file}")
            
            # 安全获取 save_dir，如果 local.py 没配置，自动使用默认路径
            env_save_dir = getattr(self.settings.env, 'save_dir', None)
            if env_save_dir is None:
                # 尝试从 workspace_dir 推断
                workspace = getattr(self.settings.env, 'workspace_dir', './')
                env_save_dir = os.path.join(workspace, 'output', 'checkpoints')
                print(f"[WARNING] 'save_dir' not set in local.py. Defaulting to: {env_save_dir}")
            
            # 确保目录存在
            full_save_path = os.path.join(env_save_dir, self.settings.project_path)
            if not os.path.exists(full_save_path):
                os.makedirs(full_save_path, exist_ok=True)
                
            print(f"Checkpoints: {full_save_path}")
            print(f"{'='*54}\n")

    def _set_default_settings(self):
        # Dict of all default values
        default = {'print_interval': 10,
                   'print_stats': None,
                   'description': ''}

        for param, default_value in default.items():
            if getattr(self.settings, param, None) is None:
                setattr(self.settings, param, default_value)

    def _clip_optimizer_grads(self):
        """Clip gradients using optimizer param groups to avoid fragile full-module traversal."""
        grad_params = []
        for group in self.optimizer.param_groups:
            for param in group.get('params', []):
                if param is not None and param.grad is not None:
                    grad_params.append(param)

        if grad_params:
            torch.nn.utils.clip_grad_norm_(grad_params, self.settings.grad_clip_norm)

    def cycle_dataset(self, loader):
        """Do a cycle of training or validation."""

        self.actor.train(loader.training)
        torch.set_grad_enabled(loader.training)

        self._init_timing()
        
        # [修改] 使用 tqdm 进度条，mininterval 防止刷新太快闪烁
        # ncols=None 让其自动适应终端宽度
        pbar = tqdm(loader, desc=f"Epoch {self.epoch} [{loader.name}]", mininterval=1.0)

        for i, data in enumerate(pbar, 1):
            if self.move_data_to_gpu:
                data = data.to(self.device)

            data['epoch'] = self.epoch
            data['settings'] = self.settings
            
            # forward pass
            if not self.use_amp:
                loss, stats = self.actor(data)
            else:
                with autocast():
                    loss, stats = self.actor(data)

            # backward pass and update weights
            if loader.training:
                self.optimizer.zero_grad()
                if not self.use_amp:
                    loss.backward()
                    if self.settings.grad_clip_norm > 0:
                        self._clip_optimizer_grads()
                    self.optimizer.step()
                else:
                    self.scaler.scale(loss).backward()
                    if self.settings.grad_clip_norm > 0:
                        self.scaler.unscale_(self.optimizer)
                        self._clip_optimizer_grads()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

            torch.cuda.synchronize()

            # update statistics
            batch_size = data['template_images'].shape[loader.stack_dim]
            self._update_stats(stats, batch_size, loader)

            # [修改] 进度条小尾巴：显示 Loss, IoU 等关键指标
            # 每 5 个 batch 刷新一次显示
            if i % 5 == 0:
                postfix = OrderedDict()
                
                # 1. 学习率
                postfix['LR'] = f"{self.optimizer.param_groups[0]['lr']:.1e}"
                
                # 安全获取数值的辅助函数
                def safe_get(key):
                    if key in stats:
                        val = stats[key]
                        return val.item() if torch.is_tensor(val) else val
                    return None

                # 2. 关键 Loss 和 指标
                if safe_get('loss/total'): postfix['Loss'] = f"{safe_get('loss/total'):.3f}"
                if safe_get('loss/giou'): postfix['Giou'] = f"{safe_get('loss/giou'):.3f}"
                if safe_get('loss/task_class'): postfix['Cls'] = f"{safe_get('loss/task_class'):.3f}"
                # 如果 Actor 返回了 IoU，则显示
                if safe_get('IoU'): postfix['IoU'] = f"{safe_get('IoU'):.3f}"

                pbar.set_postfix(postfix)

            # [修改] 写入日志文件，但 print_to_console=False 防止刷屏
            self._print_stats(i, loader, batch_size, print_to_console=False)

    def train_epoch(self):
        """Do one epoch for each loader."""
        for loader in self.loaders:
            if self.epoch % loader.epoch_interval == 0:
                if isinstance(loader.sampler, DistributedSampler):
                    loader.sampler.set_epoch(self.epoch)
                self.cycle_dataset(loader)

        self._stats_new_epoch()
        if self.settings.local_rank in [-1, 0]:
            self._write_tensorboard()

    def _init_timing(self):
        self.num_frames = 0
        self.start_time = time.time()
        self.prev_time = self.start_time

    def _update_stats(self, new_stats: OrderedDict, batch_size, loader):
        # Initialize stats if not initialized yet
        if loader.name not in self.stats.keys() or self.stats[loader.name] is None:
            self.stats[loader.name] = OrderedDict({name: AverageMeter() for name in new_stats.keys()})

        for name, val in new_stats.items():
            if name not in self.stats[loader.name].keys():
                self.stats[loader.name][name] = AverageMeter()
            self.stats[loader.name][name].update(val, batch_size)

    # [修改] 增加 print_to_console 参数
    def _print_stats(self, i, loader, batch_size, print_to_console=True):
        self.num_frames += batch_size
        current_time = time.time()
        batch_fps = batch_size / (current_time - self.prev_time)
        average_fps = self.num_frames / (current_time - self.start_time)
        self.prev_time = current_time
        
        # 只有在达到打印间隔 或 最后一个 batch 时才操作
        if i % self.settings.print_interval == 0 or i == loader.__len__():
            print_str = '[%s: %d, %d / %d] ' % (loader.name, self.epoch, i, loader.__len__())
            print_str += 'FPS: %.1f (%.1f)  ,  ' % (average_fps, batch_fps)
            for name, val in self.stats[loader.name].items():
                if (self.settings.print_stats is None or name in self.settings.print_stats):
                    if hasattr(val, 'avg'):
                        print_str += '%s: %.5f  ,  ' % (name, val.avg)

            print_str = print_str[:-5] # 去掉末尾逗号
            
            # [控制台输出] 被 cycle_dataset 的逻辑控制，防止干扰进度条
            if print_to_console:
                print(print_str)
            
            # [日志文件输出] 永远写入，不丢数据
            if misc.is_main_process():
                with open(self.settings.log_file, 'a') as f:
                    f.write(print_str + '\n')

    def _stats_new_epoch(self):
        # Record learning rate
        for loader in self.loaders:
            if loader.training:
                try:
                    lr_list = self.lr_scheduler.get_lr()
                except:
                    lr_list = self.lr_scheduler._get_lr(self.epoch)
                for i, lr in enumerate(lr_list):
                    var_name = 'LearningRate/group{}'.format(i)
                    if var_name not in self.stats[loader.name].keys():
                        self.stats[loader.name][var_name] = StatValue()
                    self.stats[loader.name][var_name].update(lr)

        for loader_stats in self.stats.values():
            if loader_stats is None:
                continue
            for stat_value in loader_stats.values():
                if hasattr(stat_value, 'new_epoch'):
                    stat_value.new_epoch()

    def _write_tensorboard(self):
        if self.epoch == 1:
            self.tensorboard_writer.write_info(self.settings.script_name, self.settings.description)

        self.tensorboard_writer.write_epoch(self.stats, self.epoch)