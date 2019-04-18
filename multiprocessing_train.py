from __future__ import division
import os

import onmt
import onmt.Markdown
import onmt.Models
import onmt.modules
import argparse
import torch
import torch.nn as nn
from torch import cuda
from multiprocess.multiprocessing_trainer import MultiprocessingTrainer
import time

parser = argparse.ArgumentParser(description='train.py')
onmt.Markdown.add_md_help_argument(parser)

# Data options

parser.add_argument('-data', required=True,
                    help='Path to the *-train.pt file from preprocess.py')
parser.add_argument('-save_model', default='model',
                    help="""Model filename (the model will be saved as
                    <save_model>_epochN_PPL.pt where PPL is the
                    validation perplexity""")
parser.add_argument('-train_from_state_dict', default='', type=str,
                    help="""If training from a checkpoint then this is the
                    path to the pretrained model's state_dict.""")
parser.add_argument('-train_from', default='', type=str,
                    help="""If training from a checkpoint then this is the
                    path to the pretrained model.""")

# Model options

parser.add_argument('-layers', type=int, default=2,
                    help='Number of layers in the LSTM encoder/decoder')
parser.add_argument('-rnn_size', type=int, default=500,
                    help='Size of LSTM hidden states')
parser.add_argument('-word_vec_size', type=int, default=500,
                    help='Word embedding sizes')
parser.add_argument('-feat_vec_size', type=int, default=20,
                    help='Feature vec sizes')
parser.add_argument('-feat_merge', type=str, default='concat',
                    choices=['concat', 'sum'],
                    help='Merge action for the features embeddings')
parser.add_argument('-feat_vec_exponent', type=float, default=0.7,
                    help="""When features embedding sizes are not set and
                    using -feat_merge concat, their dimension will be set
                    to N^feat_vec_exponent where N is the number of values
                    the feature takes""")
parser.add_argument('-input_feed', type=int, default=1,
                    help="""Feed the context vector at each time step as
                    additional input (via concatenation with the word
                    embeddings) to the decoder.""")
parser.add_argument('-rnn_type', type=str, default='LSTM',
                    choices=['LSTM', 'GRU'],
                    help="""The gate type to use in the RNNs""")
# parser.add_argument('-residual',   action="store_true",
#                     help="Add residual connections between RNN layers.")
parser.add_argument('-brnn', action='store_true',
                    help='Use a bidirectional encoder')
parser.add_argument('-brnn_merge', default='concat',
                    help="""Merge action for the bidirectional hidden states:
                    [concat|sum]""")
parser.add_argument('-copy_attn', action="store_true",
                    help='Train copy attention layer.')
parser.add_argument('-coverage_attn', action="store_true",
                    help='Train a coverage attention layer.')
parser.add_argument('-lambda_coverage', type=float, default=1,
                    help='Lambda value for coverage.')

parser.add_argument('-encoder_layer', type=str, default='rnn',
                    help="""Type of encoder layer to use.
                    Options: [rnn|mean|transformer]""")
parser.add_argument('-decoder_layer', type=str, default='rnn',
                    help='Type of decoder layer to use. [rnn|transformer]')
parser.add_argument('-context_gate', type=str, default=None,
                    choices=['source', 'target', 'both'],
                    help="""Type of context gate to use [source|target|both].
                    Do not select for no context gate.""")
parser.add_argument('-attention_type', type=str, default='general',
                    choices=['dot', 'general', 'mlp'],
                    help="""The attention type to use:
                    dotprot or general (Luong) or MLP (Bahdanau)""")

# Optimization options
parser.add_argument('-encoder_type', default='text',
                    help="Type of encoder to use. Options are [text|img].")
parser.add_argument('-batch_size', type=int, default=64,
                    help='Maximum batch size')
parser.add_argument('-max_generator_batches', type=int, default=32,
                    help="""Maximum batches of words in a sequence to run
                    the generator on in parallel. Higher is faster, but uses
                    more memory.""")
parser.add_argument('-epochs', type=int, default=100,
                    help='Number of training epochs')
parser.add_argument('-start_epoch', type=int, default=1,
                    help='The epoch from which to start')
parser.add_argument('-param_init', type=float, default=0.1,
                    help="""Parameters are initialized over uniform distribution
                    with support (-param_init, param_init).
                    Use 0 to not use initialization""")
parser.add_argument('-optim', default='sgd',
                    help="Optimization method. [sgd|adagrad|adadelta|adam]")
parser.add_argument('-max_grad_norm', type=float, default=5,
                    help="""If the norm of the gradient vector exceeds this,
                    renormalize it to have the norm equal to max_grad_norm""")
parser.add_argument('-dropout', type=float, default=0.3,
                    help='Dropout probability; applied between LSTM stacks.')
parser.add_argument('-position_encoding', action='store_true',
                    help='Use a sinusoid to mark relative words positions.')
parser.add_argument('-share_decoder_embeddings', action='store_true',
                    help='Share the word and softmax embeddings for decoder.')

parser.add_argument('-curriculum', action="store_true",
                    help="""For this many epochs, order the minibatches based
                    on source sequence length. Sometimes setting this to 1 will
                    increase convergence speed.""")
parser.add_argument('-extra_shuffle', action="store_true",
                    help="""By default only shuffle mini-batch order; when true,
                    shuffle and re-assign mini-batches""")
parser.add_argument('-truncated_decoder', type=int, default=0,
                    help="""Truncated bptt.""")

# learning rate
parser.add_argument('-learning_rate', type=float, default=1.0,
                    help="""Starting learning rate. If adagrad/adadelta/adam is
                    used, then this is the global learning rate. Recommended
                    settings: sgd = 1, adagrad = 0.1,
                    adadelta = 1, adam = 0.001""")
parser.add_argument('-learning_rate_decay', type=float, default=0.5,
                    help="""If update_learning_rate, decay learning rate by
                    this much if (i) perplexity does not decrease on the
                    validation set or (ii) epoch has gone past
                    start_decay_at""")
parser.add_argument('-start_decay_at', type=int, default=8,
                    help="""Start decaying every epoch after and including this
                    epoch""")
parser.add_argument('-start_checkpoint_at', type=int, default=0,
                    help="""Start checkpointing every epoch after and including this
                    epoch""")
parser.add_argument('-decay_method', type=str, default="",
                    help="""Use a custom learning rate decay [|noam] """)
parser.add_argument('-warmup_steps', type=int, default=4000,
                    help="""Number of warmup steps for custom decay.""")


# pretrained word vectors

parser.add_argument('-pre_word_vecs_enc',
                    help="""If a valid path is specified, then this will load
                    pretrained word embeddings on the encoder side.
                    See README for specific formatting instructions.""")
parser.add_argument('-pre_word_vecs_dec',
                    help="""If a valid path is specified, then this will load
                    pretrained word embeddings on the decoder side.
                    See README for specific formatting instructions.""")

# GPU
parser.add_argument('-gpus', default=[], nargs='+', type=int,
                    help="Use CUDA on the listed devices.")

parser.add_argument('-log_interval', type=int, default=50,
                    help="Print stats at this interval.")
parser.add_argument('-log_server', type=str, default="",
                    help="Send logs to this crayon server.")
parser.add_argument('-experiment_name', type=str, default="",
                    help="Name of the experiment for logging.")

parser.add_argument('-seed', type=int, default=2017,
                    help="""Random seed used for the experiments
                    reproducibility.""")
parser.add_argument('-add_noise', action='store_true', 
                    help="add noise to target embedding to trust less to tokens in front")
parser.add_argument('-noise', type=float, default=1.0,
                    help="add noise to target embedding to trust less to tokens in front")


#### redis
parser.add_argument('-redis', action='store_true',
                    help="use redis to load training data")
parser.add_argument('-db', type=int, default=0,
                    help="redis db")
parser.add_argument('-port', type=int, default=0,
                    help="redis port ")
parser.add_argument('-dicts', type=str, default="",
                    help="")
parser.add_argument('-reverse', action='store_true',
                    help="reverse the translation direction")
parser.add_argument('-r2l', action='store_true',
                    help="reverse target from right to left")

opt = parser.parse_args()

print(opt)

if opt.seed > 0:
    torch.manual_seed(opt.seed)

if torch.cuda.is_available() and not opt.gpus:
    print("WARNING: You have a CUDA device, should run with -gpus 0")

if opt.gpus:
    cuda.set_device(opt.gpus[0])
    if opt.seed > 0:
        torch.cuda.manual_seed(opt.seed)





def eval(trainer, criterion, data):
    stats = onmt.Loss.Statistics()
    #model.eval()
    loss = onmt.Loss.MemoryEfficientLoss(opt, None, criterion,
                                         eval=True, copy_loss=opt.copy_attn)
    for i in range(0, len(data), trainer.num_replicas):
        batchs = [data[j] for j in range(i, i+trainer.num_replicas) if j < len(data)]
        batch_stats_list = trainer.valid_step(batchs, loss)
        for batch_stats in batch_stats_list:
            stats.update(batch_stats)
    #model.train()
    return stats

def trainModel(trainer, trainData, validData, dataset):

    model_dirname = os.path.dirname(opt.save_model)
    if not os.path.exists(model_dirname):
        os.mkdir(model_dirname)
    assert os.path.isdir(model_dirname), "%s not a directory" % opt.save_model

    # Define criterion of each GPU.
    if not opt.copy_attn:
        criterion = onmt.Loss.NMTCriterion(dataset['dicts']['tgt'].size(), None)
    else:
        criterion = onmt.modules.CopyCriterion

    def trainEpoch(epoch):
        if opt.extra_shuffle and epoch > opt.curriculum:
            trainData.shuffle()

        mem_loss = onmt.Loss.MemoryEfficientLoss(opt, None,
                                                 criterion,
                                                 copy_loss=opt.copy_attn)

        # Shuffle mini batch order.
        batchOrder = torch.randperm(len(trainData))

        total_stats = onmt.Loss.Statistics()
        report_stats = onmt.Loss.Statistics()

        for i, k in enumerate(range(0, len(trainData), trainer.num_replicas)):
            #start = time.time()
            batchIdxs = batchOrder[k:k + trainer.num_replicas]
            #read_time = time.time() - start
            #print('data read index : ', batchIdx, ' used time ', read_time)
            batchs = [trainData[batchIdx] for batchIdx in batchIdxs]

            batch_stats_list = trainer.train_step(batchs, mem_loss)

            #end = time.time() - start
            #print('current index : ', batchIdx, ' used time ', end)


            for batch_stats in batch_stats_list:
                total_stats.update(batch_stats)
                report_stats.update(batch_stats)
            for batch in batchs:
                report_stats.n_src_words += batch.lengths.data.sum()

            if i % opt.log_interval == -1 % opt.log_interval:
                report_stats.output(epoch, i+1, len(trainData),
                                    total_stats.start_time)
                if opt.log_server:
                    report_stats.log("progress", opt.experiment_name, trainer.get_optimizer())
                report_stats = onmt.Loss.Statistics()

        return total_stats

    for epoch in range(opt.start_epoch, opt.epochs + 1):
        print('')


        
        #  (1) train for one epoch on the training set
        train_stats = trainEpoch(epoch)

        print('Train perplexity: %g' % train_stats.ppl())
        print('Train accuracy: %g' % train_stats.accuracy())

        #  (2) evaluate on the validation set
        valid_stats = eval(trainer, criterion, validData)
        print('Validation perplexity: %g' % valid_stats.ppl())
        print('Validation accuracy: %g' % valid_stats.accuracy())

        # Log to remote server.
        if opt.log_server:
            optim = trainer.get_optimizer()
            train_stats.log("train", opt.experiment_name, optim)
            valid_stats.log("valid", opt.experiment_name, optim)

        #  (3) update the learning rate
        trainer.lr_step(valid_stats.ppl(), epoch)


        #  (4) drop a checkpoint
        if epoch >= opt.start_checkpoint_at:
            #for rank in range(trainer.num_replicas):
            save_model(trainer, valid_stats, epoch, dataset, rank=0)


def save_model(trainer, valid_stats, epoch, dataset, rank=0):
    model = trainer.get_model(rank)
    model_state_dict = model.state_dict()
    model_state_dict = {k: v for k, v in model_state_dict.items()
                        if 'generator' not in k}
    generator_state_dict = model.generator.state_dict()
    checkpoint = {
        'model': model_state_dict,
        'generator': generator_state_dict,
        'dicts': dataset['dicts'],
        'opt': opt,
        'epoch': epoch,
        'optim': trainer.get_optimizer(rank)
    }
    torch.save(checkpoint,
               '%s.%s_acc_%.2f_ppl_%.2f_e%d.pt'
               % (opt.save_model, rank, valid_stats.accuracy(),
                  valid_stats.ppl(), epoch))


def main():
    # Set up the Crayon logging server.
    if opt.log_server != "":
        from pycrayon import CrayonClient
        cc = CrayonClient(hostname=opt.log_server)

        experiments = cc.get_experiment_names()
        print(experiments)
        if opt.experiment_name in experiments:
            cc.remove_experiment(opt.experiment_name)
        opt.experiment_name = cc.create_experiment(opt.experiment_name)

    print("Loading data from '%s'" % opt.data)

    dataset = torch.load(opt.data)
    dict_checkpoint = (opt.train_from if opt.train_from
                       else opt.train_from_state_dict)
    if dict_checkpoint:
        print('Loading dicts from checkpoint at %s' % dict_checkpoint)
        checkpoint = torch.load(dict_checkpoint,
                                map_location=lambda storage, loc: storage)
        #dataset['dicts'] = checkpoint['dicts']

    if opt.redis:
        trainData = onmt.RedisDataset("train", opt.batch_size, False, reverse=opt.reverse, port=opt.port, db=opt.db,
                                      r2l=opt.r2l)
        validData = onmt.RedisDataset('valid', opt.batch_size, False, volatile=True, reverse=opt.reverse, port=opt.port,
                                      r2l=opt.r2l, db=opt.db)
    else:
        trainData = onmt.Dataset(dataset['train']['src'],
                             dataset['train']['tgt'], opt.batch_size, False,
                             data_type=dataset.get("type", "text"),
                             srcFeatures=dataset['train'].get('src_features'),
                             tgtFeatures=dataset['train'].get('tgt_features'),
                             alignment=dataset['train'].get('alignments'))
        validData = onmt.Dataset(dataset['valid']['src'],
                             dataset['valid']['tgt'], opt.batch_size, False,
                             volatile=True,
                             data_type=dataset.get("type", "text"),
                             srcFeatures=dataset['valid'].get('src_features'),
                             tgtFeatures=dataset['valid'].get('tgt_features'),
                             alignment=dataset['valid'].get('alignments'))

    dicts = dataset['dicts']
    if opt.reverse:
        dicts['src'], dicts['tgt'] = dicts['tgt'], dicts['src']
        dicts['src_features'], dicts['tgt_features'] = dicts['tgt_features'], dicts['src_features']
    print(' * vocabulary size. source = %d; target = %d' %
          (dicts['src'].size(), dicts['tgt'].size()))
    #if 'src_features' in dicts:
    #    for j in range(len(dicts['src_features'])):
    #        print(' * src feature %d size = %d' %
    #              (j, dicts['src_features'][j].size()))

    #print(' * number of training sentences. %d' %
          #len(dataset['train']['src']))
    print(' * maximum batch size. %d' % opt.batch_size)

    print('Building model...')

    if opt.encoder_type == "text":
        encoder = onmt.Models.Encoder(opt, dicts['src'],
                                      dicts.get('src_features', None))
    elif opt.encoder_type == "img":
        encoder = onmt.modules.ImageEncoder(opt)
        assert("type" not in dataset or dataset["type"] == "img")
    else:
        print("Unsupported encoder type %s" % (opt.encoder_type))

    decoder = onmt.Models.Decoder(opt, dicts['tgt'])

    if opt.copy_attn:
        generator = onmt.modules.CopyGenerator(opt, dicts['src'], dicts['tgt'])
    else:
        generator = nn.Sequential(
            nn.Linear(opt.rnn_size, dicts['tgt'].size()),
            nn.LogSoftmax())
        if opt.share_decoder_embeddings:
            generator[0].weight = decoder.embeddings.word_lut.weight

    model = onmt.Models.NMTModel(encoder, decoder, len(opt.gpus) > 1)

    if opt.train_from:
        print('Loading model from checkpoint at %s' % opt.train_from)
        chk_model = checkpoint['model']
        generator_state_dict = chk_model.generator.state_dict()
        model_state_dict = {k: v for k, v in chk_model.state_dict().items()
                            if 'generator' not in k}
        model.load_state_dict(model_state_dict)
        generator.load_state_dict(generator_state_dict)
        opt.start_epoch = checkpoint['epoch'] + 1

    if opt.train_from_state_dict:
        print('Loading model from checkpoint at %s'
              % opt.train_from_state_dict)
        model.load_state_dict(checkpoint['model'])
        generator.load_state_dict(checkpoint['generator'])
        opt.start_epoch = checkpoint['epoch'] + 1

    model.cpu()
    generator.cpu()

    model.generator = generator

    if not opt.train_from_state_dict and not opt.train_from:
        if opt.param_init != 0.0:
            print('Intializing params')
            for p in model.parameters():
                p.data.uniform_(-opt.param_init, opt.param_init)

        encoder.embeddings.load_pretrained_vectors(opt.pre_word_vecs_enc)
        decoder.embeddings.load_pretrained_vectors(opt.pre_word_vecs_dec)

        optim = onmt.Optim(
            opt.optim, opt.learning_rate, opt.max_grad_norm,
            lr_decay=opt.learning_rate_decay,
            start_decay_at=opt.start_decay_at,
            opt=opt
        )
    else:
        print('Loading optimizer from checkpoint:')
        optim = checkpoint['optim']
        print(optim)


    if opt.train_from or opt.train_from_state_dict:
        optim.optimizer.load_state_dict(
            checkpoint['optim'].optimizer.state_dict())

    print('Multi gpu training ', opt.gpus)
    trainer = MultiprocessingTrainer(opt, model, optim, device_ids=opt.gpus)

    nParams = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % nParams)
    enc = 0
    dec = 0
    for name, param in model.named_parameters():
        if 'encoder' in name:
            enc += param.nelement()
        elif 'decoder' in name:
            dec += param.nelement()
        else:
            print(name, param.nelement())
    print('encoder: ', enc)
    print('decoder: ', dec)

    trainModel(trainer, trainData, validData, dataset)

if __name__ == "__main__":
    main()
