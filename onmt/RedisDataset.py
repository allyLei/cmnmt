from __future__ import division

import redis
import msgpack
import numpy as np

import math
import torch
from torch.autograd import Variable

import sys
from onmt.Dataset import Batch
from onmt.Constants import *


class RedisDataset(object):
    """
    """
    def __init__(self, name, batchSize, cuda, host="localhost", port=6379, db=0, reverse=False, volatile=False, r2l=False):
        """
        """
        self.cuda = cuda
        self.name = name
        self.batchSize = batchSize
        self.volatile = volatile
        self.reverse = reverse ### reverse translation direction
        #self.redis = LoadRedis(name, host, port)
        #self.redis.load(dataPath, batchSize)
        self.r = redis.Redis(host=host, port=port, db=db)
        self.r2l = r2l

        self.numBucket = int(self.r.get(self.name+"_"+"size"))
        assert self.numBucket is not None and self.numBucket > 0
        self.bucket_size = int(self.r.get("bucket_size"))
        assert self.bucket_size > 0 and self.batchSize % self.bucket_size == 0
        self.batchBucket = int(self.batchSize / self.bucket_size)
        self.numBatches = math.ceil(float(self.numBucket) / self.batchBucket)
        indices = np.random.permutation(self.numBucket).tolist()
        ##  for sanity check
        #indices = list(np.arange(self.numBucket))

        self.indices = [ name + "_" + str(indice) for indice in indices] 

    def _batchify(self, data, align_right=False,
                  include_lengths=False, dtype="text", features=None):
        if dtype == "text":
            lengths = [len(x) for x in data]
            max_length = max(lengths)
            if features:
                num_features = len(features)
                out = torch.LongTensor(len(data), max_length, num_features + 1) \
                             .fill_(PAD)
                assert (len(data) == len(features[0])), \
                    ("%s %s" % (len(data[0]), len(features[0])))
            else:
                out = torch.LongTensor(len(data), max_length) \
                             .fill_(PAD)
            for i in range(len(data)):
                data_length = len(data[i])
                offset = max_length - data_length if align_right else 0
                if features:
                    out[i].narrow(0, offset, data_length)[:, 0].copy_(torch.LongTensor(data[i]))
                    for j in range(num_features):
                        out[i].narrow(0, offset, data_length)[:, j+1] \
                             .copy_(torch.LongTensor(features[j][i]))
                    #print('num features ', num_features)
                else:
                    out[i].narrow(0, offset, data_length).copy_(torch.LongTensor(data[i]))

            if include_lengths:
                return out, lengths
            else:
                return out
        elif dtype == "img":
            heights = [x.size(1) for x in data]
            max_height = max(heights)
            widths = [x.size(2) for x in data]
            max_width = max(widths)

            out = data[0].new(len(data), 3, max_height, max_width).fill_(0)
            for i in range(len(data)):
                data_height = data[i].size(1)
                data_width = data[i].size(2)
                height_offset = max_height - data_height if align_right else 0
                width_offset = max_width - data_width if align_right else 0
                out[i].narrow(1, height_offset, data_height) \
                      .narrow(2, width_offset, data_width).copy_(data[i])
            return out, widths
    
    #@profile
    def __getitem__(self, index):
        assert index < self.numBatches, "%d > %d" %(index, self.numBatches)
        src = []
        tgt = []
        align = []
        feature = []
        #print 'batchBucket ', self.batchBucket
        s = self.batchBucket * index
        e = self.batchBucket * (index + 1)
        #print "slice"
        #print type(s), " : ", s
        #print type(e), " : ", e
        #print "---slice"
        redis_indices = self.indices[s:e]
        datas = self.r.mget(redis_indices)
        for d in datas:
            data = msgpack.unpackb(d)
            src += data[0]
            if len(data) >= 2:
                tgt_datas = data[1]
                if self.r2l:
                    reverse_tgt_datas = []
                    for tgt_data in tgt_datas:
                        main_tgt_data = tgt_data[1:-1]
                        main_tgt_data.reverse()
                        reverse_tgt_datas += [[tgt_data[0]] + main_tgt_data + [tgt_data[-1]]]
                    tgt_datas = reverse_tgt_datas
                tgt += tgt_datas
                if len(data) == 3:
                    align += data[2]
                    if len(data) == 4:
                        feature += data[3]
        batch_size = len(src)
        if len(tgt) >= 1:
            assert len(src) == len(tgt), "size of data from redis does not matches"

        features = None
        if len(feature) > 0:
            features = [feature]
        if self.reverse:
            #print "training in reverse direction"
            src, tgt = tgt, src
            align = [np.array(alignment).transpose().tolist() for alignment in align]

        srcBatch, lengths = self._batchify(src, align_right=False, include_lengths=True, dtype="text", features=features)
        if srcBatch.dim() == 2:
            srcBatch = srcBatch.unsqueeze(2)
        if tgt is not None:
            tgtBatch = self._batchify(tgt, dtype="text")
        else:
            tgtBatch = None
        alignment = None
        if len(align) > 0:
            src_len = srcBatch.size(1)
            tgt_len = tgtBatch.size(1)
            #assert src_len == len(align)
            #assert tgt_len == len(align[0])
            alignment = torch.ByteTensor(tgt_len, batch_size, src_len).fill_(0)
        #    #alignment = torch.ByteTensor(tgt_len, batch_size, src_len).fill_(0).pin_memory()
            for i in range(len(align)):
                region = torch.ByteTensor(align[i])
                alignment[1:region.size(1)+1, i, :region.size(0)] = region.t()
            alignment = alignment.float()
            if self.cuda:
                alignment = alignment.cuda()
        # tgt_len x batch x src_len
        lengths = torch.LongTensor(lengths)
        #lengths = torch.LongTensor(lengths).pin_memory()
        indices = range(len(srcBatch))
        # within batch sorting by decreasing length for variable length rnns
        lengths, perm = torch.sort(lengths, 0, descending=True)
        indices = [indices[p] for p in perm]
        srcBatch = [srcBatch[p] for p in perm]
        if tgtBatch is not None:
            tgtBatch = [tgtBatch[p] for p in perm]
        #if alignment is not None:
        #    alignment = alignment.transpose(0, 1)[
        #        perm.type_as(alignment).long()]
        #    alignment = alignment.transpose(0, 1).contiguous()

        def wrap(b, dtype="text"):
            if b is None:
                return b
            b = torch.stack(b, 0)
            if dtype == "text":
                b = b.transpose(0, 1).contiguous()
            if self.cuda:
                b = b.cuda()
            b = Variable(b, volatile=self.volatile)
            return b
        # Wrap lengths in a Variable to properly split it in DataParallel
        lengths = lengths.view(1, -1)
        lengths = Variable(lengths, volatile=self.volatile)

        return Batch(wrap(srcBatch, "text"),
                     wrap(tgtBatch, "text"),
                     lengths,
                     indices,
                     batch_size,
                     alignment=alignment)
    def __len__(self):
        return self.numBatches
if __name__ == "__main__":
    torch.cuda.set_device(2)
    test = RedisDataset("train", 128, True, volatile=False, db=1)
    import time
    BATCH = len(test)
    start = time.time()
    for idx in range(50):
        batch = test[idx]
    end = time.time() - start
    #print 'elapsed time : ',  end
