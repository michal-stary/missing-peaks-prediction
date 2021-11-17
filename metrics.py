import numpy as np
from helpers import get_mean_nan

def set_metrics(pred_next, y_next):
    pred_next_set = set(pred_next)
    y_next_set = set(y_next)
            
    TP = len(y_next_set.intersection(pred_next_set))
    FP = len(pred_next_set) - TP
    FN = len(y_next_set) - TP
    
    precision = TP/ (TP + FP) if TP+FP != 0 else np.NaN        
    recall = TP/ (TP + FN) if TP+FN != 0 else np.NaN
    
    jaccard = TP/ len(y_next_set.union(pred_next_set)) if len(y_next_set.union(pred_next_set)) != 0 else np.NaN
    
    return precision, jaccard, recall

def metrics_klj(l_pred_indices_per_k, y_indices, up_to_k=None, l=None, j=None):
    
    if up_to_k is None:
        up_to_k = len(l_pred_indices_per_k)
    else:
        up_to_k = min(up_to_k, len(l_pred_indices_per_k))
    if l is None:
        l = l_pred_indices_per_k.shape[2]
    print(f"Selected up to k={up_to_k}, l={l}, j={j}")
    
    
    precisions = np.zeros(shape=(up_to_k, l_pred_indices_per_k.shape[1]))
    jaccards = np.zeros(shape=(up_to_k, l_pred_indices_per_k.shape[1]))
    
    for k in range(up_to_k):
        for i in range(l_pred_indices_per_k.shape[1]):    
            pred_next = l_pred_indices_per_k[k][i][:l]
            y_next = y_indices[i][k:k+j]
                
            # skip too short spectra indicated by -1
            if (l_pred_indices_per_k[k][i][:l] == -1).any() or len(pred_next) != l or len(y_next) != j:
                precisions[k, i] = np.NaN
                jaccards[k, i] = np.NaN
                continue
            # calculate metrics order respecting 
            # not implemented 
            
            # calculete metrics set
            precision, jaccard, _ = set_metrics(pred_next, y_next)
            
            precisions[k, i] = precision
            jaccards[k, i] = jaccard
            
    return precisions, jaccards


def metrics_klrel(l_pred_indices_per_k, y_indices, X_intens, up_to_k=None, l=None, to_rel_inten=0.2):
    
    if up_to_k is None:
        up_to_k = len(l_pred_indices_per_k)
    else:
        up_to_k = min(up_to_k, len(l_pred_indices_per_k))
    
    print(f"Selected up to k={up_to_k}, l={l}, to_rel_inten={to_rel_inten}")
    
    
    precisions = np.zeros(shape=(up_to_k, l_pred_indices_per_k.shape[1]))
    jaccards = np.zeros(shape=(up_to_k, l_pred_indices_per_k.shape[1]))
    
    assert len(y_indices) == len(X_intens)
    
    for k in range(up_to_k):
        for i in range(l_pred_indices_per_k.shape[1]):
            #HAAAAAAAAACK
#             if len(y_indices[i]) != len(X_intens[i]):
#                 X_intens[i] = X_intens[i][:len(y_indices[i])]
#             if i > 22000:
#                 precisions[k, i] = np.NaN
#                 jaccards[k, i] = np.NaN
#                 continue
            if len(y_indices[i]) != len(X_intens[i]):
                if k == 0:
                    print("problem - spectrum with unknown peak skipped")
                precisions[k, i] = np.NaN
                jaccards[k, i] = np.NaN
                continue
            assert len(y_indices[i]) == len(X_intens[i])
            
            if len(y_indices[i]) <= k:
                precisions[k, i] = np.NaN
                jaccards[k, i] = np.NaN
                continue

                
#             if len(y_indices[i]) != len(X_intens[i]):
#                 print(len(X_intens[i]), len(y_indices[i]))
#                 print(X_intens[i], y_indices[i])
#                 print(i)
    
                
            if isinstance(X_intens[i], list):
                intens = np.array(X_intens[i])
            else:
                intens = X_intens[i]
            # print(intens)    
            # get number of peaks above some intensity of last seen peak
            j = min(np.argmax(intens < intens[k]*to_rel_inten) - k -1 , 20)
            
            # did not find any peak lower than given threshold
            if j <= -1:
                j = min(len(intens) - k, 20)
            
            # did not find any peak above or equal the given threshold
            if j == 0:
                precisions[k, i] = np.NaN
                jaccards[k, i] = np.NaN
                continue
            
            if l is None:
                curr_l = j
            else:
                curr_l = l
            # print(np.argmax(intens < intens[k]*to_rel_inten)-k)
            # print(l_pred_indices_per_k.shape)
            if l_pred_indices_per_k.shape[2] < curr_l:
                precisions[k, i] = np.NaN
                jaccards[k, i] = np.NaN

                print(curr_l)
                print("too little pred")
                continue
            
            pred_next = l_pred_indices_per_k[k][i][:curr_l]
            y_next = y_indices[i][k:k+j]    
                
            # skip spectra indicated by -1
            if (l_pred_indices_per_k[k][i][:l] == -1).any():
                precisions[k, i] = np.NaN
                jaccards[k, i] = np.NaN
                print("i")
                continue
            # calculate metrics order respecting 
            # not implemented 
            
            # calculete metrics set
            precision, jaccard, recall = set_metrics(pred_next, y_next)
            
            precisions[k, i] = precision
            jaccards[k, i] = jaccard
            
    return precisions, jaccards


def calc_mean_lj_metrics(l_pred_indices_per_k, y_indices, X_intens, up_to_k=None, \
                         l=None, j=None, l_rel=None, to_rel_inten=0.2):
    
    print(f"Possible k up to {len(l_pred_indices_per_k)}, predict up to {l_pred_indices_per_k.shape[2]} peaks")
    precisions, jaccards = metrics_klj(l_pred_indices_per_k, y_indices, up_to_k=up_to_k, l=l, j=j)
    mean_prec, mean_jac = get_mean_nan(precisions), get_mean_nan(jaccards)
    
    print((~np.isnan(precisions)).sum(axis=1))
    print((~np.isnan(jaccards)).sum(axis=1))
    precisions_at_int, jaccards_at_int = metrics_klrel(l_pred_indices_per_k, y_indices, X_intens, \
                                                       up_to_k=up_to_k, l=l_rel, to_rel_inten=to_rel_inten) 
    
    print((~np.isnan(precisions_at_int)).sum(axis=1))
    print((~np.isnan(jaccards_at_int)).sum(axis=1))
    
    
    mean_prec_int, mean_jac_int = get_mean_nan(precisions_at_int), get_mean_nan(jaccards_at_int)
    
    return mean_prec, mean_jac, mean_prec_int, mean_jac_int

def calc_mean_random_metrics(some_pred_per_m, m_pred_per_m, m_y_per_m):
    precs_m = np.zeros(shape=(len(m_y_per_m), len(m_y_per_m[0])))
    precs_some = np.zeros(shape=(len(m_y_per_m), len(m_y_per_m[0])))
    recs_some = np.zeros(shape=(len(m_y_per_m), len(m_y_per_m[0])))
    jacs_some = np.zeros(shape=(len(m_y_per_m), len(m_y_per_m[0])))
    f1_some = np.zeros(shape=(len(m_y_per_m), len(m_y_per_m[0])))

    for m in range(len(m_y_per_m)):
        for i in range(len(m_y_per_m[m])):
            # print(m_pred_per_m[m][i], m_y_per_m[m][i])
            prec, jac, recall = set_metrics(some_pred_per_m[m][i], m_y_per_m[m][i])
            
            # if m == 0:
            #     prec, jac, recall = (1,1,np.nan) if len(some_pred_per_m[m][i]) == 0 else (0,0,np.nan)
                
            precs_some[m][i] = prec
            recs_some[m][i] = recall
            jacs_some[m][i] = jac
            
            if np.isnan(prec) or np.isnan(recall) or prec+recall == 0:
                f1_some[m][i] = np.NaN
            else:
                f1_some[m][i] = 2*prec*recall/(prec+recall)
            
            prec_m, _, _ = set_metrics(m_pred_per_m[m][i], m_y_per_m[m][i])
            precs_m[m][i] = prec_m
            
    mean_prec_some = np.nanmean(precs_some, axis=1)
    mean_rec_some = np.nanmean(recs_some, axis=1)
    mean_jac_some = np.nanmean(jacs_some, axis=1)
    mean_f1_some = np.nanmean(f1_some, axis=1)
    mean_prec= np.nanmean(precs_m, axis=1)
    
    # print(mean_prec)
    # print(up_to_m)
    # HACK 
    # mean_prec_some = mean_prec
    # mean_rec_some = mean_prec
    # mean_f1_some = mean_prec
    
    return mean_prec_some, mean_rec_some, mean_jac_some, mean_f1_some, mean_prec






def accuracy_at_k(l_pred_indices_per_k, y_indices, up_to_k=None):
    if up_to_k is None:
        up_to_k = len(l_pred_indices_per_k)
    else:
        up_to_k = min(up_to_k, len(l_pred_indices_per_k))
    accs = np.zeros(up_to_k)
    
    for k in range(up_to_k):
        corr = 0
        tot = 0
        for i in range(len(y_indices)):
            
            # skip too short spectra
            if (l_pred_indices_per_k[k, i, :] == -1).any():
                continue
            # safety check 
            assert len(y_indices[i]) > k
            
            
            if l_pred_indices_per_k[k, i, 0] == y_indices[i][k]:
                corr+=1
            tot +=1
        accs[k] = 0 if tot == 0 else corr/tot 
    return accs



def accuracy_at_int(l_pred_indices_per_k, y_indices, X_intens, n_bins=10, split="uniform"):
    corr_at_int = np.zeros(n_bins)
    tot_at_int = np.zeros(n_bins)
    for k in range(len(l_pred_indices_per_k)):
        for i in range(len(y_indices)):
            
            # skip too short spectra
            if (l_pred_indices_per_k[k, i, :] == -1).any() or k>=len(X_intens[i]):
                continue
            # safety check 
            assert len(y_indices[i]) > k

            
            bin_ = get_bin(X_intens[i][k], n_bins, split)
            
            if l_pred_indices_per_k[k, i, 0] == y_indices[i][k]:
                corr_at_int[bin_] += 1
            tot_at_int[bin_] += 1
    return corr_at_int / tot_at_int





def get_bin(intensity, n_bins, split):
    if split == "uniform":
        bins = np.arange(n_bins-1)/n_bins
    return np.digitize(intensity, bins)

def metrics_intlj(l_pred_indices_per_k, y_indices, X_intens, l, j, down_to_int=None, n_bins=10, split="uniform"):
    """
    compute the precision and jaccard of peaks by agregating on the intensity of last seen peak
    """
    
    precisions = [[] for _ in range(n_bins)]
    jaccards = [[] for _ in range(n_bins)]
    
    for k in range(len(l_pred_indices_per_k)):
        for i in range(l_pred_indices_per_k.shape[1]):    
            pred_next = l_pred_indices_per_k[k][i][:l]
            y_next = y_indices[i][k:k+j]
            
            # skip too short spectra indicated by -1
            if (l_pred_indices_per_k[k][i][:l] == -1).any() or len(pred_next) != l or len(y_next) != j:
                continue
            
            # calculate metrics order respecting 
            # not implemented 
            
            # calculete metrics set
            precision, jaccard = set_metrics(pred_next, y_next)
            
            
            intensity = X_intens[i][k]
            
            bin_ = get_bin(intensity, n_bins, split)
            
            
            precisions[bin_].append(precision)
            jaccards[bin_].append(jaccard)
            
    return precisions, jaccards
