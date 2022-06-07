import os
import sys
import random
import configargparse
import numpy as np
from trainer import Trainer

def get_parser(parser=None, required=True):
    if parser is None:
        parser = configargparse.ArgumentParser(
            description="ViT Model Training on CPU or GPU",
            config_file_parser_class=configargparse.YAMLConfigFileParser,
            formatter_class = configargparse.ArgumentDefaultsHelpFormatter
        )
    parser.add("--config", is_config_file=True, help="config file path", default="conf/base.yaml")

    parser.add_argument(
        "--tag",
        type=str,
        help="Experiment Tag for storing logs, models"
    )
    parser.add_argument(
        "--seed",
        default=1,
        type=int,
        help="Random seed"
    )


    #Data Related Params
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/iemocap",
        help="Data Directory"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch Size"
    )

    #Model Related Params
    parser.add_argument(
        "--num_labels",
        type=int,
        default=4,
        help="Number of Data Labels"
    )

    #Training Related Params

    '''
    opt lr eps weight decay num_epochs
    '''
    parser.add_argument(
        "--opt",
        type=str,
        default="AdamW",
        help="Number of Data Labels"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Number of Data Labels"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Number of Data Labels"
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-8,
        help="Number of Data Labels"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Number of Data Labels"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of Data Labels"
    )
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=5
    )
    return parser

def main(cmd_args):
    parser = get_parser()
    args, _ =  parser.parse_known_args(cmd_args)
    
    random.seed(args.seed)
    np.random.seed(args.seed)

    expdir = os.path.join("exp","train_"+args.tag)
    model_dir = os.path.join(expdir,"models")
    log_dir = os.path.join(expdir,"logs")
    tb_dir = os.path.join("tensorboard","train_"+args.tag)

    args.model_dir = model_dir
    args.log_dir = log_dir
    args.tb_dir = tb_dir
    args.expdir = expdir

    for x in ["exp","tensorboard",expdir,model_dir,log_dir,tb_dir]:
        if not os.path.isdir(x):
            os.mkdir(x)
    
    trainer = Trainer(args)
    trainer.train()

if __name__ =="__main__":
    main(sys.argv[1:])
