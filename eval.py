import torch
from Parser import Parser
import logging
import os
from datetime import datetime
import torchvision.models as models
import numpy as np

import commons
import utils
import inference
import datasets_dual
import network

'''Setup'''
parser = Parser()
args = parser.parse_arguments()

commons.setup_logging(args.save_dir)
commons.seed_everything(args.seed)

utils.save_to_yaml(args)
logging.debug(f"The outputs are being saved in {args.save_dir}")

args.recall_values = list(range(1, 26))

'''Model'''
model = network.RGBTVPR_Net(pretrained_foundation = True, foundation_model_path = args.foundation_model_path)
model = model.to(args.device)

resume_path= args.resume[0]
print(f"Resuming from {resume_path}")
model = utils.resume_model(resume_path, model)

'''Dataset'''
val_ds_list = []
test_ds_list = []
test_sequences = args.test_seq

for seq in test_sequences:
    args.sequences = [seq]  
    
    val_ds0 = datasets_dual.BaseSTheReODual(args, args.datasets_folder, "test")
    val_ds_list.append(val_ds0)
    logging.info(f"[Val - {seq}] Database: {val_ds0.database_num}, Queries: {val_ds0.queries_num}, Total: {len(val_ds0)}")
    
    val_ds1 =datasets_dual.BaseSTheReODual(args, args.datasets_folder, "test")
    test_ds_list.append(val_ds1)
    logging.info(f"[Test - {seq}] Database: {val_ds1.database_num}, Queries: {val_ds1.queries_num}, Total: {len(val_ds1)}")

all_r1, all_r5 = [], []
for seq, val_dataset in zip(test_sequences, val_ds_list):
    recalls, recalls_str = inference.inference(args, val_dataset, model, seq)
    logging.info(f"Recalls on [{seq}] {val_dataset}: {recalls_str}")
    all_r1.append(recalls[0])
    all_r5.append(recalls[1])

avg_r1_r5 = np.mean(all_r1) + np.mean(all_r5)

