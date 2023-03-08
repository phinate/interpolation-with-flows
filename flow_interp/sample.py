import numpy as np
import torch


def _sample_flow(flow, feature_scaler=None, context_scaler=None, context=None, num_samples=10000, num_features=1):
    if (context is not None) and (context_scaler is not None):
        context = context_scaler.transform(context).astype("float32")
    with torch.no_grad():
        samples = flow.sample(num_samples, context=context).detach()
        
        if feature_scaler is not None:
            return feature_scaler.inverse_transform(samples.reshape(-1, feature_scaler.n_features_in_)).reshape(
            len(context), num_samples, feature_scaler.n_features_in_
        )
        else:
            return samples.reshape(
            len(context), num_samples, num_features
        )

def hist1d_from_flows_with_error(
    flow_list, bins,  context_scaler=None, context=None, num_samples=10000, density=False, return_bins=False, normalized_counts = None
):

    flow_samples = np.array([_sample_flow(flow, None, context_scaler, context, num_samples) for flow in flow_list])

    if len(context) > 1:
        hists = [
                [
                    np.histogram(data, bins=bins, density=density)[0]
                    for data, bins in zip(
                        flow_data, bins
                    )
                ]
                for flow_data in flow_samples
            ]
        
    else:
        hists = np.array(
            [
                [
                    np.histogram(
                        data, bins=bins, density=density
                    )[0]
                    for data in flow_data
                ]
                for flow_data in flow_samples
            ]
        )
    if normalized_counts is not None:
        norm_factor = normalized_counts / num_samples
    else:
        norm_factor = 1
    
    if len(hists) == 1:
        return hists[0]*norm_factor, None 
    hist_avg = np.mean(np.array(hists)*norm_factor, axis=0)  # shape: [len(truth_masses), num_samples, 2]
    hist_std = np.std(np.array(hists)*norm_factor, axis=0)  # shape: [len(truth_masses), num_samples, 2]
    return hist_avg, hist_std 


def hist2d_from_flows_with_error(
    flow_list,  x_bins, y_bins, feature_scaler, context_scaler=None, context=None, num_samples=10000, density=False, return_bins=False, normalized_counts = None
):
    flow_samples = np.array([_sample_flow(flow, feature_scaler, context_scaler, context, num_samples) for flow in flow_list])
   

    if len(context) > 1:

        if x_bins is None:
            hists = [
                    [
                        np.histogram2d(data[:, 0], data[:, 1], density=density)[0]
                        for data in flow_data
                    ]
                    for flow_data in flow_samples
                ]
        else:
            hists = [
                    [
                        np.histogram2d(data[:, 0], data[:, 1], bins=bins, density=density)[0]
                        for data, bins in zip(
                            flow_data, zip(x_bins, y_bins) 
                        )
                    ]
                    for flow_data in flow_samples
                ]
        
    else:
        if x_bins is None:
            hists = np.array(
                [
                    [
                        np.histogram2d(
                            data[:, 0], data[:, 1], density=density
                        )[0]
                        for data in flow_data
                    ]
                    for flow_data in flow_samples
                ]
            )
        else:
            hists = np.array(
                [
                    [
                        np.histogram2d(
                            data[:, 0], data[:, 1], bins=[x_bins, y_bins], density=density
                        )[0]
                        for data in flow_data
                    ]
                    for flow_data in flow_samples
                ]
            )

    if normalized_counts is not None:
        norm_factor = normalized_counts / num_samples
    else:
        norm_factor = 1
    
    if return_bins:
        bins = [
            [
                np.histogram2d(data[:, 0], data[:, 1], density=density)[1:]
                for data in flow_data
            ]
            for flow_data in flow_samples
        ]
        if len(hists) == 1:
            return hists[0]*norm_factor, None, bins
        else:
            hist_avg = np.mean(np.array(hists)*norm_factor, axis=0)  # shape: [len(truth_masses), num_samples, 2]
            hist_std = np.std(np.array(hists)*norm_factor, axis=0)  # shape: [len(truth_masses), num_samples, 2]
            return hist_avg, hist_std, bins
    if len(hists) == 1:
        return hists[0]*norm_factor, None 
    hist_avg = np.mean(np.array(hists)*norm_factor, axis=0)  # shape: [len(truth_masses), num_samples, 2]
    hist_std = np.std(np.array(hists)*norm_factor, axis=0)  # shape: [len(truth_masses), num_samples, 2]
    return hist_avg, hist_std 


def ak_to_ndarray(arr):
    return (
        arr.to_numpy()
        .astype([(field, np.float64) for field in arr.fields])
        .view((np.float64, len(arr.fields)))
    )

def make_data_hists2d(data_list, weight_list=None, num_bins = 20, bins=None, filter_IQR = False, quantiles=(0,0.2), density=True, awkward=True):
    data_hists_all = []
    bins_x = []
    bins_y = []

    for i, data in enumerate(data_list):  # loop over signal points
        if awkward:
            data_np = ak_to_ndarray(data)
        else:
            data_np = data
        x, y = data_np[:, 0], data_np[:, 1] 
        weights = weight_list[i] if weight_list is not None else None
        if filter_IQR:
            Q1 = np.quantile(data, quantiles[0])
            Q3 = np.quantile(data, quantiles[1])
            IQR = Q3 - Q1
            mask = np.any(~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))), axis=1)
            data = data[mask]
            data_bins = [np.linspace(min(x), max(x), num_bins), np.linspace(min(y), max(y), num_bins)]
            data_hists_all.append(np.histogram2d(x, y, bins=data_bins, density=density, weights=weights)[0])
        elif bins is not None:
            hist, *data_bins = np.histogram2d(x, y, density=density, weights=weights, bins=bins[i])
            data_hists_all.append(hist) 
        else:
            hist, *data_bins = np.histogram2d(x, y, density=density, weights=weights)
            data_hists_all.append(hist)
        bins_x.append(data_bins[0])
        bins_y.append(data_bins[1])


    data_hists_all = np.array(data_hists_all)
    bins_x = np.array(bins_x)
    bins_y = np.array(bins_y)
    return data_hists_all, bins_x, bins_y


def hist_metric(flow, x_bins, y_bins, feature_scaler, data_hists_all, context_points, context_points_to_check, context_scaler=None, num_samples=10000):
    # load in only those hists at points you want to eval
    # get the list of points, find the indicies that match the condition
    # load in hists/bins, mask them accordingly
    mask = [point in context_points_to_check for point in context_points]
    data_hists = data_hists_all[mask]
    bins_x_masked = x_bins[mask]
    bins_y_masked = y_bins[mask]
    # sample from flow
    hists, _ = hist2d_from_flows_with_error(
        [flow],  bins_x_masked, bins_y_masked, feature_scaler, context_scaler=context_scaler, context=context_points_to_check, num_samples=num_samples, density=True
    )
    hists = np.array(hists)
    assert hists.shape == data_hists.shape
    return np.mean((hists-data_hists)**2)