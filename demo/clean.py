import torch
import sys
sys.path.append('/home/lixiaopu/work/py/OpenNMT')
import argparse

args = argparse.ArgumentParser()
args.add_argument('-model', type=str, required=True,
                help="")
args.add_argument('-save', type=str, required=True,
                help="")

opt = args.parse_args()


model = torch.load(opt.model)

keys = ["optim", "epoch"]

new_model = {k:v for k,v in model.items() if k not in keys}

torch.save(new_model, opt.save)
