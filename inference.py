import faiss
import torch
import logging
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
# from datetime import datetime
import time

# TODO: can be less memory cost
# TODO: finish the uncompleted parts
def inference(args, eval_ds, model, pca=None, k=1, use_cuda=True, verbose=True):
    '''
    hard_resize: directly use the resized image
    single_query: use the resized image, and set query_infer_batchsize=1 (used when the query images have varying size)
    central_crop: Take the biggest central crop of size self.resize. Preserves ratio.
    five_crops: use five crops of the image, and take the average of the features
    nearest_crop: use five crops of the image, 
    maj_voting: calculate features of five crops of the image, then use the nearest features
    * hard_size method for all database images
    * selected test_method for all query images
    '''

    model = model.eval()
    with torch.no_grad():
        # Extract database features
        start_time = time.time()
        database_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num)))
        database_dataloader = DataLoader(dataset=database_subset_ds, num_workers=args.num_workers,
                                        batch_size=args.infer_batch_size, prefetch_factor=1,
                                        pin_memory=(args.device=="cuda"))
        database_features = np.empty((eval_ds.database_num, args.features_dim), dtype="float32")
        
        for inputs, indices in tqdm(database_dataloader, ncols=100):
            flags = torch.zeros(inputs.shape[0], dtype=torch.bool)
            features = model(inputs.to(args.device), flags).view(-1, args.features_dim)
            features = features.cpu().numpy()
            database_features[indices.numpy(), :] = features

        logging.info(f"Finished extracting {eval_ds.database_num} database features in {time.time() - start_time:.2f} s")

        # Extract query features
        start_time = time.time()
        queries_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num, len(eval_ds))))
        queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args.num_workers,
                                        batch_size=args.infer_batch_size, pin_memory=(args.device=="cuda"))
        queries_features = np.empty((eval_ds.queries_num, args.features_dim), dtype="float32")

        for inputs, indices in tqdm(queries_dataloader, ncols=100):
            flags = torch.ones(inputs.shape[0], dtype=torch.bool).to(args.device)
            features = model(inputs.to(args.device), flags).view(-1, args.features_dim)
            features = features.cpu().numpy()
            queries_features[indices.numpy()-eval_ds.database_num, :] = features

        logging.info(f"Finished extracting {eval_ds.queries_num} query features in {time.time() - start_time:.2f} s")
    
    faiss_index = faiss.IndexFlatL2(args.features_dim)
    faiss_index.add(database_features)
    del database_features

    ### Calculating recalls
    start_time = time.time()
    
    distances, predictions = faiss_index.search(queries_features, max(args.recall_values))
    del queries_features

    # For each query, check if the predictions are correct
    positives_per_query = eval_ds.get_positives()
    
    # args.recall_values by default is [1, 5, 10, 20]
    recalls = np.zeros(len(args.recall_values))
    for query_index, pred in enumerate(predictions):
        for i, n in enumerate(args.recall_values):
            if np.any(np.in1d(pred[:n], positives_per_query[query_index])):
                recalls[i:] += 1
                break
            
    # Divide by the number of queries*100, so the recalls are in percentages
    recalls = recalls / eval_ds.queries_num * 100
    logging.info(f"recalls: {','.join(map(str, recalls))}")
    recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(args.recall_values, recalls)])
    
    return recalls, recalls_str