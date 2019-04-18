# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#

"""
Train a network on multiple GPUs using multiprocessing.
"""

import torch
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau

from multiprocess import nccl
from .multiprocessing_event_loop import MultiprocessingEventLoop, Future
#from .multiprocessing_pdb import pdb
import onmt

from time import clock
class Timer(object):
    def __init__(self, verbose=False):
        self.verbose = verbose
    def __enter__(self):
        self.start = clock()
        return self
    def __exit__(self, *args):
        self.end = clock()
        self.secs = self.end - self.start
        self.msecs = self.secs * 1000  # millisecs
        if self.verbose:
            print('elapsed time: %f ms' % self.msecs)

class MultiprocessingTrainer(MultiprocessingEventLoop):
    """Main class for multi-GPU training.

    Each GPU has a full copy of the model and is assigned to its own Python
    process. Gradients are accumulated with all-reduce and all model replicas
    are updated synchronously after each batch.

    The methods in this class are divided into synchronous functions, which
    prepare and dispatch the input to each process, and asynchronous functions
    (prefixed with `_async_`), which run on each process in parallel.
    """

    def __init__(self, args, model, optimizer, device_ids=None,
                 multiprocessing_method='spawn'):
        if device_ids is None:
            device_ids = tuple(range(torch.cuda.device_count()))
        super().__init__(device_ids, multiprocessing_method)

        if not torch.cuda.is_available():
            raise NotImplementedError('Training on CPU is not supported')
        model = model.share_memory()
        nccl_uid = nccl.get_unique_id()

        Future.gen_list([
            self.call_async(rank, '_async_init', args=args, model=model, optimizer=optimizer,
                            nccl_uid=nccl_uid)
            for rank in range(self.num_replicas)
        ])

    def _async_init(self, rank, device_id, args, model, optimizer, nccl_uid):
        """Initialize child processes."""
        self.args = args

        # set torch.seed in this process
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

        # set CUDA device
        torch.cuda.set_device(device_id)

        # initialize NCCL
        nccl.initialize(self.num_replicas, nccl_uid, rank)

        # copy model to current device
        self.model = model.cuda()

        # initialize optimizer
        self.optimizer = optimizer
        self.optimizer.set_parameters(self.model.parameters())
        self.flat_grads = None


    def get_model(self, rank=0):
        """Get one of the model replicas."""
        # just return the first model, since all replicas are the same
        return self.call_async(rank, '_async_get_model').gen()

    def _async_get_model(self, rank, device_id):
        return self.model

    def get_optimizer(self, rank=0):
        return self.call_async(rank, '_async_get_optimizer').gen()

    def _async_get_optimizer(self,rank, device_id):
        return self.optimizer

    def train_step(self, samples, criterion):
        """Do forward, backward and gradient step in parallel."""

        # scatter sample across GPUs
        self._scatter_samples(samples)

        # forward pass, backward pass and gradient step
        batch_stats = [
            self.call_async(rank, '_async_train_step', criterion=criterion)
            for rank in range(self.num_replicas)
        ]

        # aggregate losses and gradient norms
        batch_stats_list = Future.gen_list(batch_stats)

        return batch_stats_list

    def _async_train_step(self, rank, device_id, criterion):
        #with Timer() as t0:
        self.model.train()

        # zero grads even if net_input is None, since we will all-reduce them
        self.model.zero_grad()

        # calculate loss and grads
        batch_stats = onmt.Loss.Statistics()
        if self._sample is not None:
            target_size = self._sample.tgt.size(0)
            dec_state = None
            # trunc_size = self.args.truncated_decoder if self.args.truncated_decoder else target_size
            trunc_size = target_size
            # for j in range(0, target_size - 1, trunc_size):
            j = 0
            trunc_batch = self._sample.truncate(j, j + trunc_size)
            #with Timer() as t1:
            outputs, attn, dec_state = self.model(trunc_batch.src, trunc_batch.tgt, trunc_batch.lengths, dec_state)
            #print("rank", rank, "forward time:", t1.msecs)
            #with Timer() as t2:
            batch_stats, inputs, grads = criterion.loss(trunc_batch, outputs, attn, self.model.generator)
            #print("rank", rank, "loss time:", t2.msecs)
            #with Timer() as t3:
            torch.autograd.backward(inputs, grads)
            #print("rank", rank, "backward time:", t3.msecs)

            # if dec_state is not None:
            #    dec_state.detach()

        # flatten grads into a contiguous block of memory
        if self.flat_grads is None:
            self.flat_grads = self._flatten_grads_(self.model)

        #self.flat_grads /= self.num_replicas
        #with Timer() as t4:
        nccl.all_reduce(self.flat_grads)
            #pass
        #print("rank", rank, "all_reduce time:", t4.msecs)
        # grad_norm = self._clip_grads_(self.flat_grads, self.args.clip_norm)
        #with Timer() as t5:
        self.optimizer.step()
        #print("rank", rank, "optimize time:", t5.msecs)
        #print("rank:",rank,"all train step time:", t0.msecs)

        return batch_stats

    def _flatten_grads_(self, model):
        num_params = sum(p.data.numel() for p in model.parameters())
        flat_grads = next(model.parameters()).data.new(num_params)
        offset = 0
        for p in model.parameters():
            grad = p.grad.data
            numel, sz = grad.numel(), grad.size()
            flat_grads[offset:offset+numel] = grad.view(-1)
            grad.set_(flat_grads[offset:offset+numel])
            grad.resize_(sz)  # preserve original shape
            offset += numel
        return flat_grads

    def _clip_grads_(self, flat_grads, clipv):
        norm = flat_grads.norm()
        if clipv > 0 and norm > clipv:
            coef = max(norm, 1e-6) / clipv
            flat_grads.div_(coef)
        return norm

    def valid_step(self, samples, criterion):
        """Do forward pass in parallel."""
        # scatter sample across GPUs
        self._scatter_samples(samples, volatile=True)
        #criterion.prepare(samples)

        # forward pass
        batch_stats_list = [
            self.call_async(rank, '_async_valid_step', criterion=criterion)
            for rank in range(self.num_replicas)
        ]

        # aggregate losses
        #loss = criterion.aggregate(Future.gen_list(losses))
        batch_stats_list = Future.gen_list(batch_stats_list)

        return batch_stats_list

    def _async_valid_step(self, rank, device_id, criterion):
        batch_stats = onmt.Loss.Statistics()
        if self._sample is None:
            return batch_stats

        self.model.eval()
        outputs, attn, dec_state = self.model(self._sample.src, self._sample.tgt, self._sample.lengths)
        batch_stats, inputs, grads = criterion.loss(self._sample, outputs, attn, self.model.generator)

        #net_output = self.model(**self._sample['net_input'])
        #loss = criterion(net_output, self._sample)

        return batch_stats

    def get_lr(self):
        """Get the current learning rate."""
        return self.call_async(0, '_async_get_lr').gen()

    def _async_get_lr(self, rank, device_id):
        return self.optimizer.param_groups[0]['lr']

    def lr_step(self, val_ppl=None, epoch=None):
        """Adjust the learning rate depending on the validation loss."""
        lr = Future.gen_list([
            self.call_async(rank, '_async_lr_step', val_ppl=val_ppl, epoch=epoch)
            for rank in range(self.num_replicas)
        ])
        return lr[0]

    def _async_lr_step(self, rank, device_id, epoch, val_ppl):
        self.optimizer.updateLearningRate(val_ppl, epoch)
        return self.optimizer.optimizer.param_groups[0]['lr']

    def _scatter_samples(self, samples, volatile=False):
        """Split and distribute a sample across GPUs."""
        # Pad with None until its size is equal to the number of replicas.
        samples = samples + [None]*(self.num_replicas - len(samples))

        Future.gen_list([
            self.call_async(rank, '_async_prepare_sample', sample=samples[rank], volatile=volatile)
            for rank in range(self.num_replicas)
        ])

    def _async_prepare_sample(self, rank, device_id, sample, volatile):
        if sample is None:
            self._sample = None
        else:
            sample.src = sample.src.cuda(device_id)
            sample.tgt = sample.tgt.cuda(device_id)
            self._sample = sample
