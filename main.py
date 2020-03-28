import os
import sys
import pickle
import signal
import argparse
import traceback

from sentence_transformers import SentenceTransformer
import torch
import numpy as np

from dataloader import loader,parallel_sampler
import train.factory as train_utils
from classifier.factory import  get_classifier

def parse_args():
    parser = argparse.ArgumentParser(
            description="Few Shot nlp tasks With Pretrained Embeddings")

    # data configuration
    parser.add_argument("--data_path", type=str,
                        default="data/event.json",
                        help="path to dataset")
    parser.add_argument("--dataset", type=str, default="event",
                        help="name of the dataset. "
                        "Options: [event]")
    
    parser.add_argument("--n_workers", type=int, default=48,
                        help="Num. of cores used for loading data. Set this "
                        "to zero if you want to use all the cpus.")
    
    # task configuration
    parser.add_argument("--way", type=int, default=1,
                        help="#classes for each task")
    parser.add_argument("--shot", type=int, default=40,
                        help="#support examples for each class for each task")
    parser.add_argument("--query", type=int, default=25,
                        help="#query examples for each class for each task")
    parser.add_argument("--subclass", type=int, default=2,
                        help="#subclasses for each class")

    # train/test configuration
    parser.add_argument("--train_epochs", type=int, default = 1000,
                        help="max num of training epochs")
    parser.add_argument("--train_episodes", type=int, default=100,
                        help="#tasks sampled during each training epoch")
    parser.add_argument("--val_episodes", type=int, default=100,
                        help="#tasks sampled during each validation epoch")
    parser.add_argument("--test_episodes", type=int, default=1000,
                        help="#tasks sampled during each testing epoch")
    
    # model options
    parser.add_argument("--embedding", type=str, default="bert",
                        help=("document embedding method. Options: "
                              "[bert]"))
    parser.add_argument("--classifier", type=str, default="r2d2",
                        help=("classifier. Options: [proto,r2d2]"))
    
    
    # proto configuration
    parser.add_argument("--proto_hidden", nargs="+", type=int,
                        default=[300,300],
                        help=("hidden dimension of the proto-net"))
  
    
    # training options
    parser.add_argument("--seed", type=int, default=330, help="seed")
    parser.add_argument("--dropout", type=float, default=0.2 ,help="drop rate") #0.1
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate") #1e-3
    parser.add_argument("--patience", type=int, default=10, help="patience") #从20改成了10
    parser.add_argument("--clip_grad", type=float, default=None,
                        help="gradient clipping")
    parser.add_argument("--cuda", type=int, default=-1,
                        help="cuda device, -1 for cpu")
    parser.add_argument("--save", action="store_true", default=False,
                        help="train the model")
    parser.add_argument("--notqdm", action="store_true", default=False,
                        help="disable tqdm")
    parser.add_argument("--result_path", type=str, default="")
    parser.add_argument("--snapshot", type=str, default="",
                        help="path to the pretraiend weights")

    return parser.parse_args()

def print_args(args):
    """
        Print arguments (only show the relevant arguments)
    """
    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))
        
def set_seed(seed):
    """
        Setting random seeds
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
        
def main():
    args = parse_args()
    if args.dataset == 'event':
        args.way = 1
    print_args(args)
    set_seed(args.seed)
    train_data, val_data, test_data= loader.load_dataset(args.dataset,args.data_path,args)
    ebd_model = None
    ebd_dim = 0
    ebd_path = 'model/bert-base-nli-stsb-mean-tokens'
    if args.embedding == 'bert':
        if args.cuda == -1:
            ebd_model = SentenceTransformer(ebd_path).cpu()
        elif args.cuda == 0:
            ebd_model = SentenceTransformer(ebd_path)
        else:
            ebd_model = SentenceTransformer(
                ebd_path).cuda(args.cuda)
        ebd_dim = 768
    
    model = {}
    model["ebd"] = ebd_model
    model["clf"] = get_classifier(args,ebd_dim)

    train_utils.train(train_data, val_data, model,args)
    test_acc, test_std = train_utils.test(test_data, model,args,args.test_episodes)
    
    if args.result_path:
        directory = args.result_path[:args.result_path.rfind("/")]
        if not os.path.exists(directory):
            os.mkdirs(directory)

        result = {
            "test_acc": test_acc,
            "test_std": test_std,
            "val_acc": val_acc,
            "val_std": val_std
        }

        for attr, value in sorted(args.__dict__.items()):
            result[attr] = value

        with open(args.result_path, "wb") as f:
            pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)
            
if __name__ == "__main__":
    try:
        main()
    except Exception:
        exc_info = sys.exc_info()
        traceback.print_exception(*exc_info)
        os.killpg(0, signal.SIGKILL)

    exit(0)

            
    

    




    

