import os
import math
import torch
import logging
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import multiprocessing
from datetime import datetime
from torch.utils.data.dataloader import DataLoader

from Parser import Parser
from utils import *
import commons
import datasets_dual
import inference
import network

def train(args, start_time):
    '''Datasets'''
    DATASET_FOLDER = "./Datasets"
    ############################################################
    args.sequences = args.train_seq
    
    triplets_ds = datasets_dual.TripletsSTheReODual(args, DATASET_FOLDER)
    logging.info(f"Train query set: {triplets_ds}")
    logging.info(f"[Train - {args.train_seq}] Database: {triplets_ds.database_num}, Queries: {triplets_ds.queries_num}")

    test_sequences = args.test_seq
    val_ds_list = []
    test_ds_list = []

    for seq in test_sequences:
        args.sequences = [seq]  
        
        val_ds0 = datasets_dual.BaseSTheReODual(args, args.datasets_folder, "test")
        val_ds_list.append(val_ds0)
        logging.info(f"[Val - {seq}] Database: {val_ds0.database_num}, Queries: {val_ds0.queries_num}, Total: {len(val_ds0)}")
        
        val_ds1 =datasets_dual.BaseSTheReODual(args, args.datasets_folder, "test")
        test_ds_list.append(val_ds1)
        logging.info(f"[Test - {seq}] Database: {val_ds1.database_num}, Queries: {val_ds1.queries_num}, Total: {len(val_ds1)}")

    args.sequences = args.train_seq
    ############################################################

    '''Model'''
    model = network.RGBTVPR_Net(pretrained_foundation = True, foundation_model_path = args.foundation_model_path)
    model = model.to(args.device)
    model = torch.nn.DataParallel(model)

    ## Freeze parameters except adapter
    for name, param in model.module.rgb_backbone.named_parameters():
        if "adapter" not in name:
            param.requires_grad = False
            
    for name, param in model.module.thermal_backbone.named_parameters():
        if "adapter" not in name:
            param.requires_grad = False

    ## initialize Adapter
    for n, m in model.named_modules():
        if 'adapter' in n:
            for n2, m2 in m.named_modules():
                if 'D_fc2' in n2:
                    if isinstance(m2, nn.Linear):
                        nn.init.constant_(m2.weight, 0.)
                        nn.init.constant_(m2.bias, 0.)
            for n2, m2 in m.named_modules():
                if 'conv' in n2:
                    if isinstance(m2, nn.Conv2d):
                        nn.init.constant_(m2.weight, 0.00001)
                        nn.init.constant_(m2.bias, 0.00001)

    '''Optimizer'''
    if args.optim == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    elif args.optim == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.001)

    '''Loss Function'''
    GlobalTriplet = nn.TripletMarginLoss(margin=args.margin, p=2, reduction="sum")

    '''Resume from checkpoint'''
    if args.resume:
        model, _, best_r1_r5, start_epoch_num, not_improved_num = utils.resume_train(args, model, strict=False)
        logging.info(f"Resuming from epoch {start_epoch_num} with best (R@1 + R@5) {best_r1_r5:.1f}")
    else:
        best_r1_r5 = start_epoch_num = not_improved_num = 0

    # Flags
    thermal_flag = torch.ones(1, dtype=torch.bool)
    rgb_flags = torch.zeros(1 + args.negs_num_per_query, dtype=torch.bool)
    bundle_flags = torch.cat([thermal_flag, rgb_flags]).to(args.device)  
    query_flags = bundle_flags.repeat(args.train_batch_size)

    '''Training'''
    for epoch_num in range(start_epoch_num, args.epochs_num):
        logging.info(f"Start training epoch: {epoch_num:02d}")

        epoch_start_time = datetime.now()
        epoch_losses = np.zeros((0, 1), dtype=np.float32)

        # How many loops should an epoch last (default is 5000/1000=5)
        loops_num = math.ceil(args.queries_per_epoch / args.cache_refresh_rate)
        for loop_num in range(loops_num):
            logging.debug(f"Cache: {loop_num + 1} / {loops_num}")

            # Compute triplets to use in the triplet loss
            triplets_ds.is_inference = True
            triplets_ds.compute_triplets(args, model)
            triplets_ds.is_inference = False

            logging.debug("Finish computing triplets")

            triplets_dl = DataLoader(dataset=triplets_ds, num_workers=args.num_workers,
                                    batch_size=args.train_batch_size,
                                    collate_fn=datasets_dual.collate_fn,
                                    pin_memory=(args.device == "cuda"),
                                    prefetch_factor=1,
                                    drop_last=True)
            model = model.train()

            logging.debug(f"Start loading {len(triplets_ds)} triplets as {len(triplets_dl)} batches")

            for images, triplets_local_indexes, _ in tqdm(triplets_dl, ncols=100):
                global_features = model(images.to(args.device), query_flags)

                triplets_local_indexes = torch.transpose(
                    triplets_local_indexes.view(args.train_batch_size, args.negs_num_per_query, 3), 1, 0)
                
                global_loss = 0.0
                for triplets in triplets_local_indexes:
                    queries_indexes, positives_indexes, negatives_indexes = triplets.T

                    global_loss += GlobalTriplet(global_features[queries_indexes],
                                                global_features[positives_indexes],
                                                global_features[negatives_indexes])
                global_loss /= (args.train_batch_size * args.negs_num_per_query)

                optimizer.zero_grad()
                global_loss.backward()
                optimizer.step()

                batch_loss = global_loss.item()
                epoch_losses = np.append(epoch_losses, batch_loss)
                
                del global_loss, global_features
                if args.use_fast_track: break

            logging.info(f"Epoch[{epoch_num:02d}]({loop_num + 1}/{loops_num}): " +
                        f"current batch triplet loss = {batch_loss:.8f}, " +
                        f"average epoch triplet loss = {epoch_losses.mean():.8f}")
            if args.use_fast_track: break

        logging.info(f"epoch {epoch_num:02d} time: {str(datetime.now() - epoch_start_time)[:-7]}, ")

        # Compute recalls on all validation sequences
        all_r1, all_r5 = [], []
        for seq, val_dataset in zip(test_sequences, val_ds_list):
            recalls, recalls_str = inference.inference(args, val_dataset, model, seq)
            logging.info(f"Recalls on [{seq}] {val_dataset}: {recalls_str}")
            all_r1.append(recalls[0])
            all_r5.append(recalls[1])

        avg_r1_r5 = np.mean(all_r1) + np.mean(all_r5)
        is_best = avg_r1_r5 > best_r1_r5

        save_checkpoint(args, {"epoch_num": epoch_num, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "recalls": (np.mean(all_r1), np.mean(all_r5)), "best_r1_r5": best_r1_r5,
            "not_improved_num": not_improved_num
        }, is_best, filename="last_model.pth")

        if is_best:
            logging.info(f"Improved: previous best avg (R@1 + R@5) = {best_r1_r5:.1f}, current = {avg_r1_r5:.1f}")
            best_r1_r5 = avg_r1_r5
            not_improved_num = 0
        else:
            not_improved_num += 1
            logging.info(f"Not improved: {not_improved_num} / {args.patience}: best = {best_r1_r5:.1f}, current = {avg_r1_r5:.1f}")
            if not_improved_num >= args.patience:
                logging.info(f"Performance did not improve for {not_improved_num} epochs. Stop training.")
                break
            
    logging.info(f"Best avg (R@1 + R@5): {best_r1_r5:.1f}")
    logging.info(f"Trained for {epoch_num+1:02d} epochs, in total in {str(datetime.now() - start_time)[:-7]}")

if __name__ == "__main__":
    '''Setup'''
    parser = Parser()
    args = parser.parse_arguments()

    '''Save Logs'''
    args.save_dir = os.path.join(args.save_dir, args.comment, get_timestamp())
    commons.setup_logging(args.save_dir)
    save_files(args.save_dir)
    save_to_yaml(args)
    
    logging.debug(f"The outputs are being saved in {args.save_dir}")
    logging.info(f"Use {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs")
    
    '''Start Train'''
    start_time = datetime.now()
    commons.seed_everything(args.seed)
    train(args, start_time)