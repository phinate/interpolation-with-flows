import torch
import numpy as np 
import copy
from functools import partial
from torch.optim.lr_scheduler import StepLR

def batches(train, batch_size, rng_state=0):
    num_train = train.shape[0]
    num_complete_batches, leftover = divmod(num_train, batch_size)
    num_batches = num_complete_batches + bool(leftover)

    # batching mechanism
    def data_stream():
        rng = npr.RandomState(rng_state)
        while True:
            perm = rng.permutation(num_train)
            for i in range(num_batches):
                batch_idx = perm[i * batch_size : (i + 1) * batch_size]
                yield train[batch_idx]

    return data_stream()
def train_loop(num_iter, flow, optimizer, batch_iterator, feature_scaler=None, use_weights = False, valid_set = None, context_scaler=None, metric = None, print_every = 100, scheduler=None, scheduler_hpars = None):
    
    
    if (scheduler is not None) and (scheduler_hpars is not None):
        scheduler = scheduler(optimizer, **scheduler_hpars)
    best_metric = 99999
    best_flow = None
    best_iter = 0
    losses = [], []
    if valid_set is not None:
        if use_weights:
            valid_features, valid_context, valid_weights = valid_set
        else: 
            valid_features, valid_context = valid_set

        if feature_scaler is not None:
            valid_features = torch.tensor(feature_scaler.transform(valid_features), dtype=torch.float32)
        valid_inputs = torch.tensor(valid_features, dtype=torch.float32)
        if use_weights:
            valid_weights = torch.tensor(valid_weights, dtype=torch.float32)
        else:
            valid_weights = 1
        if valid_context is not None:
            valid_context = torch.tensor(context_scaler.transform(valid_context), dtype=torch.float32)

    for i in range(num_iter):
        if use_weights:
            features, context, weights = next(batch_iterator)
            weights = torch.tensor(weights, dtype=torch.float32)
        else:
            features, context = next(batch_iterator)
            weights = 1

        if feature_scaler is not None:
            features = torch.tensor(feature_scaler.transform(features), dtype=torch.float32)
        inputs = torch.tensor(features, dtype=torch.float32)

        if context is not None:
            context = torch.tensor(context_scaler.transform(context), dtype=torch.float32)

        optimizer.zero_grad()
#         print(inputs.shape, context.shape, weights.shape)
        loss = torch.mean(-flow.log_prob(inputs=inputs, context=context)*weights)
        if torch.any(torch.isnan(loss)):
            print("nans")
            return 'nans', None, None, None
#         losses[0].append(loss)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if valid_set is not None:
            with torch.no_grad():
                valid_loss = torch.mean(-flow.log_prob(inputs=valid_inputs, context=valid_context)*valid_weights)
#                 losses[1].append(valid_loss) 
                if metric is not None:
                    valid_metric = metric(flow, valid_inputs, valid_context, valid_weights)
                else:
                    valid_metric = valid_loss
                if valid_metric < best_metric:
                    best_metric = valid_metric
                    best_flow = copy.deepcopy(flow.state_dict())
                    best_iter = i
        if (i == 0) or ((i+1) % print_every == 0):
            print(f'iteration {i}:')
            print(f'train loss = {loss}')
            if metric is not None:
                print(f'train metric = {metric(flow, inputs, context, weights)}')
            if valid_set is not None:
                print(f'valid loss = {valid_loss}')
                if metric is not None:
                    print(f'valid metric = {metric(flow, valid_inputs, valid_context, valid_weights)}')

    return best_flow, best_metric, best_iter, losses