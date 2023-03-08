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
    flow_list, bins, context_scaler=None, context=None, num_samples=10000, density=False
):
    flow_samples = np.array([_sample_flow(flow, None, context_scaler, context, num_samples) for flow in flow_list])

    if (len(np.array(x_bins).shape)) > 1:
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
    if len(hists) == 1:
        return hists[0], None 
    hist_avg = np.mean(hists, axis=0)  # shape: [len(truth_masses), num_samples]
    hist_std = np.std(hists, axis=0)  # shape: [len(truth_masses), num_samples]
    return hist_avg, hist_std
        
        
def hist2d_from_flows_with_error(
    flow_list,  x_bins, y_bins, feature_scaler, context_scaler=None, context=None, num_samples=10000, density=False
):
    flow_samples = np.array([_sample_flow(flow, feature_scaler, context_scaler, context, num_samples) for flow in flow_list])

    if (len(np.array(x_bins).shape)) > 1:
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
    if len(hists) == 1:
        return hists[0], None 
    hist_avg = np.mean(hists, axis=0)  # shape: [len(truth_masses), num_samples, 2]
    hist_std = np.std(hists, axis=0)  # shape: [len(truth_masses), num_samples, 2]
    return hist_avg, hist_std


def ak_to_ndarray(arr):
    return (
        arr.to_numpy()
        .astype([(field, np.float64) for field in arr.fields])
        .view((np.float64, len(arr.fields)))
    )

def make_data_hists2d(data_list, weight_list=None, num_bins = 20, bins = None, filter_IQR = False, quantiles=(0,0.2), signal_list=None):
    data_hists_all = []
    bins_x = []
    bins_y = []

    for i, data in enumerate(data_list):  # loop over signal points
        x, y = data[:, 0], data[:, 1] 
        weights = weight_list[i] if weight_list is not None else None
#         if filter_IQR:
#             Q1 = np.quantile(data, quantiles[0])
#             Q3 = np.quantile(data, quantiles[1])
#             IQR = Q3 - Q1
#             mask = np.any(~((data < (Q1 - 1 * IQR)) | (data > (Q3 + 1 * IQR))), axis=1)
#             data = data[mask]
#             x2, y2 = data[:, 0], data[:, 1] 
#             bins = [np.linspace(min(x2), max(x2), num_bins), np.linspace(min(y2), max(y2), num_bins)]
#             data_hists_all.append(np.histogram2d(x, y, bins=bins, density=True, weights=weights)[0])
#         elif bins is not None:
#             hist, *bins = np.histogram2d(x, y, density=True, bins = bins, weights=weights)
#             data_hists_all.append(hist)
        if signal_list is not None:
            context = signal_list[i]
            nbins = num_bins/2
            xlim = (context[0]-(nbins*20), context[0] + (nbins*20))
            ylim = (context[1]-(nbins*20), context[1] + (nbins*20))
            bins = [np.linspace(*xlim, num_bins+1), np.linspace(*ylim, num_bins+1)]
            hist, *_ = np.histogram2d(x, y, density=True, weights=weights, bins=bins)
            data_hists_all.append(hist)
#         else:
#             hist, *_ = np.histogram2d(x, y, density=True, weights=weights)
#             data_hists_all.append(hist)
        bins_x.append(bins[0])
        bins_y.append(bins[1])


    data_hists_all = np.array(data_hists_all)
    bins_x = np.array(bins_x)
    bins_y = np.array(bins_y)
    return data_hists_all, bins_x, bins_y


def hist_metric(flows, x_bins, y_bins, feature_scaler, data_hists_all, context_points, context_points_to_check, context_scaler=None, num_samples=10000):
    # load in only those hists at points you want to eval
    # get the list of points, find the indicies that match the condition
    # load in hists/bins, mask them accordingly
    mask = [point in context_points_to_check for point in context_points]
    data_hists = data_hists_all[mask]
    bins_x_masked = x_bins[mask]
    bins_y_masked = y_bins[mask]
    # sample from flow
    hists, _ = hist2d_from_flows_with_error(
        flows,  bins_x_masked, bins_y_masked, feature_scaler, context_scaler=context_scaler, context=context_points_to_check, num_samples=num_samples, density=True
    )
    hists = np.array(hists)
    assert hists.shape == data_hists.shape
    return np.mean((hists-data_hists)**2)