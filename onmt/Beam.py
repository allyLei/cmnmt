from __future__ import division
import torch
import onmt

import pdb

"""
 Class for managing the internals of the beam search process.


         hyp1-hyp1---hyp1 -hyp1
                 \             /
         hyp2 \-hyp2 /-hyp2hyp2
                               /      \
         hyp3-hyp3---hyp3 -hyp3
         ========================

 Takes care of beams, back pointers, and scores.
"""


class Beam(object):
    def __init__(self, size, cuda=False, global_scorer=None, alpha=1.0, beta=0.0, tgtDict=None):
        self.tgtDict = tgtDict
        self.pruning = {}
        if self.tgtDict is not None:
            prue = [u'！', u'!', u'。', u'.', u'？',u'?', u'；', u';', u'，', u',']
            for tok in prue:
                val = tgtDict.lookup(tok)
                if val is not None:
                    self.pruning[tok] = val
        self.size = size
        self.eosTop = False

        self.alpha = alpha
        self.beta = beta

        self.n_best = 20
        self.globalScorer = global_scorer

        self.globalState = {}
        self.tt = torch.cuda if cuda else torch

        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        self.allScores = []

        # The backpointers at each time-step.
        self.prevKs = []

        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size).fill_(onmt.Constants.PAD)]
        self.nextYs[0][0] = onmt.Constants.BOS

        # The attentions (matrix) for each time.
        self.attn = []

        self.finished = []

    def getCurrentState(self):
        "Get the outputs for the current timestep."
        return self.nextYs[-1]

    def getCurrentOrigin(self):
        "Get the backpointers for the current timestep."
        return self.prevKs[-1]

    def advance(self, wordLk, attnOut):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attnOut`: Compute and update the beam search.

        Parameters:

        * `wordLk`- probs of advancing from the last step (K x words)
        * `attnOut`- attention at the last step

        Returns: True if beam search is complete.
        """
        numWords = wordLk.size(1)

        #print "attn size", attnOut.size()

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] == onmt.Constants.EOS:
                    beamLk[i] = -1e20
                #### prune the case of punctuation after punctuation
                if self.nextYs[-1][i] in self.pruning.values():
                    for val in self.pruning.values():
                        beamLk[i][val] = -1e20
        else:
            beamLk = wordLk[0]

        flatBeamLk = beamLk.view(-1)

        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)
        self.allScores.append(self.scores)
        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = bestScoresId / numWords ###  the beam index
        self.prevKs.append(prevK)
        self.nextYs.append(bestScoresId - prevK * numWords) ### the word index in the beam
        self.attn.append(attnOut.index_select(0, prevK))


        if self.globalScorer is not None:
            self.globalScorer.updateGlobalState(self)

        for i in range(self.nextYs[-1].size(0)):
            if self.nextYs[-1][i] == onmt.Constants.EOS:
                ### 
                normalize_ratio = (5 + len(self.nextYs) -1) / 6.0
                s = self.scores[i] / (normalize_ratio ** self.alpha)
                #print "init score", s
                #s = self.scores[i] / (float(len(self.nextYs)-1) ** self.alpha)
                #scores = self.scores.clone().div_(float(len(self.nextYs)-1))
                #s = scores[i]
                if self.globalScorer is not None:
                    globalScores = self.globalScorer.score(self, self.scores)
                    s = globalScores[i]
                #print "google score", s
                self.finished.append((self.scores[i], s, len(self.nextYs)-1, i))
        # End condition is when top-of-beam is EOS.
        if self.nextYs[-1][0] == onmt.Constants.EOS:
            #self.done = True
            self.eosTop = True
            self.allScores.append(self.scores)
        return self.done()

    def done(self):
        
        #return len(self.finished) >= self.n_best
        return self.eosTop and len(self.finished) >= self.n_best

    def sortBest(self):
        return torch.sort(self.scores, 0, True)

    def sortFinished(self):
        self.finished.sort(key=lambda a: -a[1])
        scores = [(raw_sc, sc) for raw_sc, sc, _, _ in self.finished]
        ks = [(t, k) for _, _, t, k in self.finished]
        #print('there are ', len(self.finished), ' hyps in the poo')
        #print "size : ", self.globalState["coverage"].size()
        return scores, ks

    def getBest(self):
        "Get the score of the best in the beam."
        scores, ids = self.sortBest()
        #return scores[0], ids[0]
        return scores[1], ids[1]


    def getHyp(self, timestep, k):
        """
        Walk back to construct the full hypothesis.

        Parameters.

             * `k` - the position in the beam to construct.

         Returns.

            1. The hypothesis
            2. The attention at each time step.
        """
        hyp, attn = [], []
        for j in range(len(self.prevKs[:timestep]) - 1, -1, -1):
            hyp.append(self.nextYs[j+1][k])
            attn.append(self.attn[j][k])
            k = self.prevKs[j][k]

        return hyp[::-1], torch.stack(attn[::-1])

class GNMTGlobalScorer(object):
    """
    Google NMT ranking score from Wu et al.
    """
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def score(self, beam, logprobs):
        "Additional term add to log probability"
        cov = beam.globalState["coverage"]
        cov.add_(cov.eq(0).float().mul_(1e5))
        pen = self.beta * torch.min(cov, cov.clone().fill_(1.0)).log().sum(1)
        l_term = ((5.0 + len(beam.nextYs)) / (5.0 + 1.0 ) ) ** self.alpha

        if self.beta == 0.0:
            return logprobs / l_term
        return (logprobs / l_term) + pen

    def updateGlobalState(self, beam):
        "Keeps the coverage vector as sum of attens"
        if len(beam.prevKs) == 1:
            beam.globalState["coverage"] = beam.attn[-1]
        else:
            beam.globalState["coverage"] = beam.globalState["coverage"] \
                .index_select(0, beam.prevKs[-1]).add(beam.attn[-1])
        #print  beam.globalState["coverage"]
