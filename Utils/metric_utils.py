import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# compute NLL
def nll(labels, probs):
    flat_probs = probs  # probs.reshape([-1, nlabels])
    flat_labels = labels  # labels.reshape([len(flat_probs)])
    plabel = flat_probs[np.arange(len(flat_labels)), flat_labels]
    plabel[plabel == 0] = 1e-8
    return -np.log(plabel)

# compute entropy
def ent_k(p):
    # N x K
    eps = 1e-8
    return -np.sum(p * np.log(p+eps), axis=1)

# compute accuracy per bin
def compute_accuracy(conf_thresh_lower, conf_thresh_upper, conf, pred, true):
    filtered_tuples = [x for x in zip(pred, true, conf) if x[2] > conf_thresh_lower and x[2] <= conf_thresh_upper]
    if len(filtered_tuples) < 1:
        return 0, 0, 0
    else:
        correct = len([x for x in filtered_tuples if x[0] == x[1]])
        avg_conf = sum([x[2] for x in filtered_tuples]) / len(filtered_tuples)
        accuracy = float(correct) / len(filtered_tuples)
        perc_of_data = float(len(filtered_tuples)) / len(conf)
        return accuracy, perc_of_data, avg_conf


# calibration evaluation
def reliability_diagrams(predictions, truths, confidences, bin_size=0.1):
    upper_bounds = np.arange(bin_size, 1 + bin_size, bin_size)
    accs = []

    # Compute empirical probability for each bin
    plot_x = []
    perc_bins = []
    for conf_thresh in upper_bounds:
        acc, perc_pred, avg_conf = compute_accuracy(conf_thresh - bin_size, conf_thresh, confidences, predictions,
                                                    truths)
        plot_x.append(avg_conf)
        accs.append(acc)
        perc_bins.append(perc_pred)
    fig = None
    # fig.savefig("reliability.tif", format='tif', bbox_inches='tight', dpi=1200)
    # fig.savefig("reliability.eps", format='eps', bbox_inches='tight', dpi=1200)

    #     plt.show()
    return fig, plot_x, accs, perc_bins

def reliability_diagrams_plot(predictions, truths, confidences, bin_size=0.1):
    upper_bounds = np.arange(bin_size, 1 + bin_size, bin_size)
    accs = []

    # Compute empirical probability for each bin
    plot_x = []
    perc_bins = []
    for conf_thresh in upper_bounds:
        acc, perc_pred, avg_conf = compute_accuracy(conf_thresh - bin_size, conf_thresh, confidences, predictions,
                                                    truths)
        plot_x.append(avg_conf)
        accs.append(acc)
        perc_bins.append(perc_pred)

    sns.set(font_scale=1)
    # sns.set_palette("bright")
    fig, ax = plt.subplots()


    ax.bar(upper_bounds, accs, width=0.08, color='blue', edgecolor='blue', label="output")
    ax.bar(upper_bounds, upper_bounds-accs, bottom=accs, width=0.08, color='red', edgecolor='red',
           hatch='/', alpha=0.3, label="gap")
    # ax.plot(plot_x, accs, '-o', label="Accuracy", color="blue")
    ax.set_ylim([0, 1])
    ax.set_xlim([0, 1.1])
    plt.ylabel('Empirical accuracy')
    plt.xlabel('Estimated probability')
    plt.set_cmap('RdBu')
    ECE = 100 * np.sum(np.array(perc_bins) * np.abs(np.array(plot_x) - np.array(accs)))
    ax.text(0.7, 0.1, 'ECE = {0:.3g}%'.format(ECE), fontsize=14,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax.plot(plot_x, plot_x, '-.', color="red", linewidth=2) # label="Gold Standard",
    ax.legend()
    fig.set_size_inches(5, 5)
    return fig, plot_x, accs, perc_bins

"""
    implementation of OOD metrics, see reference in calculate_log.py
    from repo https://github.com/pokaxpoka/deep_Mahalanobis_detector
"""

def get_curve(in_stats, out_stats, stypes=['prob', 'ent_k']):
    tp, fp = dict(), dict()
    tnr_at_tpr95 = dict()
    for stype in stypes:
        known = in_stats[stype]
        novel = out_stats[stype]
        known.sort()
        novel.sort()
        end = np.max([np.max(known), np.max(novel)])
        start = np.min([np.min(known), np.min(novel)])
        num_k = known.shape[0]
        num_n = novel.shape[0]
        tp[stype] = -np.ones([num_k + num_n + 1], dtype=int)
        fp[stype] = -np.ones([num_k + num_n + 1], dtype=int)
        tp[stype][0], fp[stype][0] = num_k, num_n
        k, n = 0, 0
        for l in range(num_k + num_n):
            if k == num_k:
                tp[stype][l + 1:] = tp[stype][l]
                fp[stype][l + 1:] = np.arange(fp[stype][l] - 1, -1, -1)
                break
            elif n == num_n:
                tp[stype][l + 1:] = np.arange(tp[stype][l] - 1, -1, -1)
                fp[stype][l + 1:] = fp[stype][l]
                break
            else:
                if novel[n] < known[k]:
                    n += 1
                    tp[stype][l + 1] = tp[stype][l]
                    fp[stype][l + 1] = fp[stype][l] - 1
                else:
                    k += 1
                    tp[stype][l + 1] = tp[stype][l] - 1
                    fp[stype][l + 1] = fp[stype][l]
        tpr95_pos = np.abs(tp[stype] / num_k - .95).argmin()
        tnr_at_tpr95[stype] = 1. - fp[stype][tpr95_pos] / num_n
    return tp, fp, tnr_at_tpr95

def ood_metric(in_stats, out_stats, stypes=['prob', 'ent_k'], verbose=False):
    tp, fp, tnr_at_tpr95 = get_curve(in_stats, out_stats, stypes)
    results = dict()
    mtypes = ['TNR', 'AUROC', 'DTACC', 'AUIN', 'AUOUT']
    if verbose:
        print('      ', end='')
        for mtype in mtypes:
            print(' {mtype:6s}'.format(mtype=mtype), end='')
        print('')

    for stype in stypes:
        if verbose:
            print('{stype:5s} '.format(stype=stype), end='')
        results[stype] = dict()

        # TNR
        mtype = 'TNR'
        results[stype][mtype] = tnr_at_tpr95[stype]
        if verbose:
            print(' {val:6.3f}'.format(val=100. * results[stype][mtype]), end='')

        # AUROC
        mtype = 'AUROC'
        tpr = np.concatenate([[1.], tp[stype] / tp[stype][0], [0.]])
        fpr = np.concatenate([[1.], fp[stype] / fp[stype][0], [0.]])
        results[stype][mtype] = -np.trapz(1. - fpr, tpr)
        if verbose:
            print(' {val:6.3f}'.format(val=100. * results[stype][mtype]), end='')

        # DTACC
        mtype = 'DTACC'
        results[stype][mtype] = .5 * (tp[stype] / tp[stype][0] + 1. - fp[stype] / fp[stype][0]).max()
        if verbose:
            print(' {val:6.3f}'.format(val=100. * results[stype][mtype]), end='')

        # AUIN
        mtype = 'AUIN'
        denom = tp[stype] + fp[stype]
        denom[denom == 0.] = -1.
        pin_ind = np.concatenate([[True], denom > 0., [True]])
        pin = np.concatenate([[.5], tp[stype] / denom, [0.]])
        results[stype][mtype] = -np.trapz(pin[pin_ind], tpr[pin_ind])
        if verbose:
            print(' {val:6.3f}'.format(val=100. * results[stype][mtype]), end='')

        # AUOUT
        mtype = 'AUOUT'
        denom = tp[stype][0] - tp[stype] + fp[stype][0] - fp[stype]
        denom[denom == 0.] = -1.
        pout_ind = np.concatenate([[True], denom > 0., [True]])
        pout = np.concatenate([[0.], (fp[stype][0] - fp[stype]) / denom, [.5]])
        results[stype][mtype] = np.trapz(pout[pout_ind], 1. - fpr[pout_ind])
        if verbose:
            print(' {val:6.3f}'.format(val=100. * results[stype][mtype]), end='')
            print('')

    return results
