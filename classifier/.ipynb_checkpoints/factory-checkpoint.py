import torch
from classifier.proto import PROTO
from classifier.r2d2 import R2D2
from dataloader.utils import tprint


def get_classifier(args,ebd_dim):
    tprint("Building classifier")
    model = None
    if args.classifier == 'proto':
        model = PROTO(ebd_dim,args)
    if args.classifier == 'r2d2':
        model = R2D2(ebd_dim,args)
    if args.cuda != -1:
        return model.cuda(args.cuda)
    else:
        return model
