import onmt
import onmt.Models
import onmt.modules
import onmt.IO
import torch.nn as nn
import torch
from torch.autograd import Variable


class Translator(object):
    def __init__(self, opt):
        self.opt = opt
        self.tt = torch.cuda if opt.cuda else torch
        self.beam_accum = None

        checkpoints = [torch.load(model, map_location=lambda storage, loc: storage) for model in opt.models]

        model_opts = [checkpoint['opt'] for checkpoint in checkpoints]

        #### sanity check
        #model_opt.add_noise = False
        self.src_dict = checkpoints[0]['dicts']['src']
        self.tgt_dict = checkpoints[0]['dicts']['tgt']
        self.align = self.src_dict.align(self.tgt_dict)
        self.src_feature_dicts = checkpoints[0]['dicts'].get('src_features', None)
        self._type = "text"
        #self._type = model_opt.encoder_type if "encoder_type" in model_opt else "text"

        self.copy_attn = "std"
        #self.copy_attn = model_opt.copy_attn if "copy_attn" in model_opt else "std"

        if self._type == "text":
            encoders = [onmt.Models.Encoder(model_opt, self.src_dict, self.src_feature_dicts) for model_opt in model_opts]
        elif self._type == "img":
            encoders = onmt.modules.ImageEncoder(model_opt)

        self.r2l = model_opts[0].r2l if "r2l" in model_opts[0] else False
        decoders = [onmt.Models.Decoder(model_opt, self.tgt_dict) for model_opt in model_opts]
        models = [onmt.Models.NMTModel(encoder, decoder) for encoder, decoder in zip(encoders, decoders)]
        if not self.copy_attn or self.copy_attn == "std":
            generators = [nn.Sequential(
                nn.Linear(model_opt.rnn_size, self.tgt_dict.size()),
                nn.LogSoftmax()) for model_opt in model_opts]
        elif self.copy_attn:
            generator = onmt.modules.CopyGenerator(model_opt, self.src_dict,
                                                   self.tgt_dict)

        self.models = []
        for model, generator, checkpoint in zip(models, generators, checkpoints):
            model.load_state_dict(checkpoint['model'])
            generator.load_state_dict(checkpoint['generator'])
            if opt.cuda:
                model.cuda()
                generator.cuda()
            else:
                model.cpu()
                generator.cpu()
            model.generator = generator
            model.eval()
            self.models.append(model)

    def initBeamAccum(self):
        self.beam_accum = {
            "predicted_ids": [],
            "predicted_labels": [],
            "beam_parent_ids": [],
            "scores": [],
            "log_probs": [],
            'finished' : []}

    def buildData(self, srcBatch, goldBatch):
        srcFeats = []
        if self.src_feature_dicts:
            srcFeats = [[] for i in range(len(self.src_feature_dicts))]
        srcData = []
        tgtData = []
        for b in srcBatch:
            _, srcD, srcFeat = onmt.IO.readSrcLine(b, self.src_dict,
                                                   self.src_feature_dicts,
                                                   self._type)
            srcData += [srcD]
            for i in range(len(srcFeats)):
                srcFeats[i] += [srcFeat[i]]

        if goldBatch:
            for b in goldBatch:
                _, tgtD, tgtFeat = onmt.IO.readTgtLine(b, self.tgt_dict,
                                                       None, self._type)
                tgtData += [tgtD]

        return onmt.Dataset(srcData, tgtData, self.opt.batch_size,
                            self.opt.cuda, volatile=True,
                            data_type=self._type,
                            srcFeatures=srcFeats)

    def buildTargetTokens(self, pred, src, attn):
        tokens = self.tgt_dict.convertToLabels(pred, onmt.Constants.EOS)
        tokens = tokens[:-1]  # EOS
        if self.opt.replace_unk:
            for i in range(len(tokens)):
                if tokens[i] == onmt.Constants.UNK_WORD:
                    _, maxIndex = attn[i].max(0)
                    try:
                        tokens[i] = src[maxIndex[0]]
                    except IndexError:
                        print('index error ', i)
        if self.r2l:
            tokens.reverse()
        return tokens

    def translateBatch(self, batch):
        beamSize = self.opt.beam_size
        batchSize = batch.batchSize

        #  (1) run the encoder on the src
        useMasking = (self._type == "text")
        encStatesL = []
        decStatesL = []
        contextL = []
        src_lengths = batch.lengths.data.view(-1).tolist()
        globalScorer = onmt.GNMTGlobalScorer(self.opt.alpha, self.opt.beta)
        beam = [onmt.Beam(beamSize, self.opt.cuda, globalScorer, alpha=self.opt.alpha, beta=self.opt.beta, tgtDict=self.tgt_dict) for i in range(batchSize)]
        for model in self.models:
            encStates, context = model.encoder(batch.src, lengths=batch.lengths)
            encStates = model.init_decoder_state(context, encStates)

            decoder = model.decoder
            attentionLayer = decoder.attn

            #  This mask is applied to the attention model inside the decoder
            #  so that the attention ignores source padding
            padMask = batch.words().data.eq(onmt.Constants.PAD).t()
            attentionLayer.applyMask(padMask)
            #  (2) if a target is specified, compute the 'goldScore'
            #  (i.e. log likelihood) of the target under the model
            
            ## for sanity check
            #if batch.tgt is not None:
            #    decStates = encStates
            #    mask(padMask.unsqueeze(0))
            #    decOut, decStates, attn = self.model.decoder(batch.tgt[:-1],
            #                                                 batch.src,
            #                                                 context,
            #                                                 decStates)
            #    for dec_t, tgt_t in zip(decOut, batch.tgt[1:].data):
            #        gen_t = self.model.generator.forward(dec_t)
            #        tgt_t = tgt_t.unsqueeze(1)
            #        scores = gen_t.data.gather(1, tgt_t)
            #        scores.masked_fill_(tgt_t.eq(onmt.Constants.PAD), 0)
            #        goldScores += scores
            # for sanity check

            #  (3) run the decoder to generate sentences, using beam search
            # Each hypothesis in the beam uses the same context
            # and initial decoder state
            context = Variable(context.data.repeat(1, beamSize, 1))
            contextL.append(context.clone())
            goldScores = context.data.new(batchSize).zero_()
            decStates = encStates
            decStates.repeatBeam_(beamSize)
            decStatesL.append(decStates)
        batch_src = Variable(batch.src.data.repeat(1, beamSize, 1))
        padMask = batch.src.data[:, :, 0].eq(onmt.Constants.PAD).t() \
                                   .unsqueeze(0) \
                                   .repeat(beamSize, 1, 1)

        #  (3b) The main loop
        beam_done = []
        for i in range(self.opt.max_sent_length):
            # (a) Run RNN decoder forward one step.
            #mask(padMask)

            input = torch.stack([b.getCurrentState() for b in beam])\
                         .t().contiguous().view(1, -1)
            input = Variable(input, volatile=True)
            decOutTmp = []
            attnTmp = []
            word_scores = []
            for idx in range(len(self.models)):
                model = self.models[idx]
                model.decoder.attn.applyMask(padMask)
                decOut, decStatesTmp, attn = model.decoder(input, batch_src, contextL[idx], decStatesL[idx])
                decStatesL[idx] = decStatesTmp
                decOutTmp.append(decOut)
                attnTmp.append(attn)
                decOut = decOut.squeeze(0)
                # decOut: (beam*batch) x numWords
                attn["std"] = attn["std"].view(beamSize, batchSize, -1) \
                                     .transpose(0, 1).contiguous()

                # (b) Compute a vector of batch*beam word scores.
                #if not self.copy_attn:
                if True:
                    out = model.generator[0].forward(decOut)
                    out = nn.Softmax()(out)
                else:
                    # Copy Attention Case
                    words = batch.words().t()
                    words = torch.stack([words[i] for i, b in enumerate(beam)])\
                                 .contiguous()
                    attn_copy = attn["copy"].view(beamSize, batchSize, -1) \
                                            .transpose(0, 1).contiguous()

                    out, c_attn_t \
                        = self.model.generator.forward(
                            decOut, attn_copy.view(-1, batch_src.size(0)))

                    for b in range(out.size(0)):
                        for c in range(c_attn_t.size(1)):
                            v = self.align[words[0, c].data[0]]
                            if v != onmt.Constants.PAD:
                                out[b, v] += c_attn_t[b, c]
                    out = out.log()

                #score = out.view(beamSize, batchSize, -1).transpose(0, 1).contiguous()
                # batch x beam x numWords
                word_scores.append(out.clone())
            word_score = torch.stack(word_scores).sum(0).squeeze(0).div_(len(self.models))
            mean_score = word_score.view(beamSize, batchSize, -1).transpose(0, 1).contiguous()

            scores = torch.log(mean_score) 
            #scores = self.models[0].generator[1].forward(mean_score)

            # (c) Advance each beam.
            active = []

            for b in range(batchSize):
                if b in beam_done:
                    continue
                beam[b].advance(scores.data[b],
                                          attn["std"].data[b])
                is_done = beam[b].done()
                if not is_done:
                    active += [b]
                for dec in decStatesL: 
                    dec.beamUpdate_(b, beam[b].getCurrentOrigin(), beamSize)
                if is_done:
                    beam_done.append(b)
            #if not active:
                #break
            if len(beam_done) == batchSize:
                break

        #  (4) package everything up
        allHyp, allScores, allAttn = [], [], []
        n_best = self.opt.n_best

        for b in range(batchSize):
            scores, ks = beam[b].sortFinished()
            #scores, ks = beam[b].sortBest()

            allScores += [scores[:n_best]]
            hyps, attn = [], []
            for i, (times, k) in enumerate(ks[:n_best]):
                hyp, att = beam[b].getHyp(times, k)
                hyps.append(hyp)
                attn.append(att)
            allHyp += [hyps]
            if useMasking:
                valid_attn = batch.src.data[:, b, 0].ne(onmt.Constants.PAD) \
                                                .nonzero().squeeze(1)
                attn = [a.index_select(1, valid_attn) for a in attn]
            allAttn += [attn]

            # For debugging visualization.
            if self.beam_accum:
                self.beam_accum["beam_parent_ids"].append(
                    [t.tolist()
                     for t in beam[b].prevKs])
                self.beam_accum["scores"].append([
                    ["%4f" % s for s in t.tolist()]
                    for t in beam[b].allScores][1:])
                self.beam_accum["predicted_ids"].append(
                    [[idx for idx in t.tolist()]
                     for t in beam[b].nextYs][1:])
                self.beam_accum["predicted_labels"].append(
                    [[self.tgt_dict.getLabel(idx)
                      for idx in t.tolist()]
                     for t in beam[b].nextYs][1:])
                beam[b].finished.sort(key=lambda a:-a[0])
                self.beam_accum['finished'].append(beam[b].finished)

        return allHyp, allScores, allAttn, goldScores

    def translate(self, srcBatch, goldBatch):
        #  (1) convert words to indexes
        dataset = self.buildData(srcBatch, goldBatch)
        batch = dataset[0] ### first batch and the only batch
        batchSize = batch.batchSize


        #  (2) translate
        pred, predScore, attn, goldScore = self.translateBatch(batch)
        pred, predScore, attn, goldScore = list(zip(
            *sorted(zip(pred, predScore, attn, goldScore, batch.indices),
                    key=lambda x: x[-1])))[:-1]

        #  (3) convert indexes to words
        predBatch = []
        for b in range(batchSize):
            if len(pred[b]) == 0:
                predBatch.append([['translation hyp empty']])
            else:
                out_n_best = self.opt.n_best if self.opt.n_best < len(pred[b]) else len(pred[b])
                predBatch.append([self.buildTargetTokens(pred[b][n], srcBatch[b], attn[b][n]) for n in range(out_n_best)])
        return predBatch, predScore, goldScore, attn, batch.src
