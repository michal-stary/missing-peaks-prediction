import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
from helpers import get_top_k_ind
import matplotlib.colors as mcolors

COLORS = [
'tab:blue',
'tab:orange',
'tab:green',
'tab:red',
'tab:purple',
'tab:brown',
'tab:pink',
'tab:gray',
'tab:olive',
'tab:cyan',
]
def plot_scores(scores, metrics="mpi", x=None, grouper_f=lambda x: "x", \
                orderer_f =lambda x: x, title=None, save_to_path=None,\
                xlabel=None, ylabel=None, hue_f=None):
    #sns.set()
    plt.figure(figsize=(7,4))
    styles= ["-", "--",  ":", "-."]
    map_ = {"":-1}
    # print(scores.keys())
    for p_name in sorted(scores.keys(), key=orderer_f):
    
        kind = grouper_f(p_name)
        # print(hue_f(p_name))
        if kind not in map_:
            map_[kind] = max(map_.values())+1
        kind = map_[kind]
            
        #plt.plot(scores[p_name][metrics])
        if x is None:
            x=np.arange(len(scores[p_name][metrics]))+1
        if hue_f is not None:
            sns.lineplot(y=scores[p_name][metrics], x=x, linestyle=styles[kind], \
                         linewidth=.7, color=COLORS[hue_f(p_name)])
        else:
             sns.lineplot(y=scores[p_name][metrics], x=x, linestyle=styles[kind], \
                         linewidth=.7)

    plt.legend(sorted(scores.keys(), key=orderer_f))
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.ylim(0,1)
    plt.grid(color = 'whitesmoke', linestyle = '-')

    if save_to_path is not None:
        plt.savefig(save_to_path, bbox_inches='tight')
    plt.show()
    
def plot_training(learner):
    plt.figure(figsize=(8, 8))
    plt.plot(learner.train_losses)
    plt.plot(learner.val_losses)
    plt.legend(["training loss", "validation loss"])
    plt.title(f"{learner.model_name} - loss over epochs")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    #plt.ylim(1,2)
    plt.show()
    
    
def plot_stats(data1D, baseline1D=None, max_len=None, title=None, log_y=False, color="blue", \
               decreasing=False, ylim=None, ylabel=None, xlabel=None, \
               x_factor=1, disable_scientific=False):
    sns.set()
    plt.figure(figsize=(20,10))
    if title:
        plt.title(title)
    if max_len is None:
        max_len = len(data1D)
        
    x = np.arange(max_len)*x_factor
    
    
    if baseline1D is not None:
        ax = sns.scatterplot(y=baseline1D[:max_len], x=x, color="darkgrey")  
        ax = sns.scatterplot(y=data1D[:max_len], x=x, color=color, ax=ax)
    else:
        ax = sns.scatterplot(y=data1D[:max_len], x=x, color=color)
    
    
    if decreasing:
        plt.gca().invert_xaxis()
    if log_y:
        ax.set(yscale="log")
        
    if ylim is not None:
        ax.set_ylim(ylim)
    if disable_scientific:
        ax.ticklabel_format(style='plain')
    
    ax.set(ylabel=ylabel, xlabel=xlabel)
    plt.xticks(ticks=np.arange(0,max_len,10)*x_factor,fontsize=8, rotation=45)
    plt.show()    

    
def change_width(ax, new_value) :
    for patch in ax.patches :
        current_width = patch.get_width()
        diff = current_width - new_value

        # we change the bar width
        patch.set_width(new_value)

        # we recenter the bar
        patch.set_x(patch.get_x() + diff * .5)
    
def plot_spectrum_predictions(ref_doc, k, prob, coder, n_detail=10, plot_full=True, \
                              log_y=False, save_to_path=None):
    """
    Parameters
    ----------
        ds - torch dataset from which to take the samples
        model - trained nn.Module model
        n_detail - number of most intensive peaks with detailed m/z location 
        index - index of sample in the dataset ds
    """
    matplotlib.rc_file_defaults()

    
    # basic variables
    n_dec = ref_doc.n_decimals
    bars_len = coder.max_mz #len(base)
    base = np.arange(bars_len)
    p_inten = ref_doc.weights
    
    ######################
    # reference spectrum #
    ######################
    
    # create an y axis array of bars
    bars = np.zeros(bars_len)
    for p, peak in enumerate(ref_doc.words):
        #loc = peakname_to_loc(peak, n_dec)
        loc = coder.text_peak_to_mz(peak, n_dec)
        bars[loc] = max(bars[loc],p_inten[p])
      
    # get top_k_ind
    top_k_ind, _ = get_top_k_ind(ref_doc.peaks.intensities, k)
    
    
    # distinguish top_k peaks by color
    hue = np.repeat("missing peak", bars_len)
    for ind in top_k_ind:
        hue[coder.text_peak_to_mz(ref_doc.words[ind], n_dec)] = f"regular peak"
        
    # distinguish filtered by color
    hue_pred = np.repeat("returned", bars_len)
    for ind in top_k_ind:
        hue_pred[coder.text_peak_to_mz(ref_doc.words[ind], n_dec)] = "filtered"
    palette ={"filtered": "rosybrown", "returned": "crimson", 
              "missing peak": "tab:orange", "regular peak": "tab:blue"}
    
    if plot_full:
        # plot it
        f, (ax1, ax2) = plt.subplots(nrows=2, ncols=1,figsize=(40, 10), sharex=True)
        ax1 = sns.barplot(x=base, y=bars, hue=hue, ax=ax1)

    
    # add labels on top of n_details most intensive peaks    
    max_l_ref = np.sort(bars)[-n_detail]
    big_ref = np.nonzero( bars >= max_l_ref)[0]
    
    if plot_full:
        plt.sca(ax1)
        for loc in big_ref:
            plt.text(loc, bars[loc]+np.max(bars)/20, loc, ha='center', rotation=90, va='bottom')

    
    #########################
    # predicted distributon #
    #########################
    
    
    # convert prediction from spec2vec indexing to location on mz indexing
    bars_pred = coder.transform_to_mz(prob, n_dec)
    
    
    # add labels on top of n_details most probable peaks
    max_l = np.sort(bars_pred)[-n_detail]
    big = np.nonzero( bars_pred >= max_l)[0]    
    
    if plot_full:
        plt.sca(ax2)
        for loc in big:
            plt.text(loc, bars_pred[loc]+np.max(bars_pred)/20, loc, ha='center', rotation=90, va='bottom')

        # plot it
        ax2 = sns.barplot(x=base, y=bars_pred, ax=ax2, dodge=False, hue=hue_pred, palette=palette)
        change_width(ax2, .75)    
  
        
        if log_y:
            ax1.set(yscale="log")
            ax2.set(yscale="log")
            
            ax2.set_ylim(10e-5)
            ax1.sharey(ax2)
        
        ax1.title.set_text(f"Full reference spectrum of {ref_doc.metadata['name']}")
        ax2.title.set_text(f"Predicted distribution for {len(top_k_ind)+1}. peak")
        ax1.xaxis.set_tick_params(labelbottom=True)
        ax1.set(ylabel='intensity')
        ax2.set(xlabel='m/z', ylabel='prediction')
        plt.sca(ax1)
        plt.xticks(ticks=np.arange(0,1001,10),fontsize=8, rotation=45)
        plt.sca(ax2)
        plt.xticks(ticks=np.arange(0,1001,10), fontsize=8, rotation=45)
        plt.show()

    ##################################
    # focused wiew on the dense area #
    ##################################
    
    # get focus area with np.quantile and nasty hack
    meta = []
    for peak, intensity in zip(ref_doc.words, p_inten):
        meta += [coder.text_peak_to_mz(peak, n_dec)]*int(intensity*1000)
    focus = max(0, np.quantile(meta, 0.05) -50), np.quantile(meta, 0.95) + 50
    
    
    #plot focused 
    f, (ax1, ax2) = plt.subplots(nrows=2, ncols=1,figsize=(20, 10), sharex=True)
    ax1 = sns.barplot(x=base, y=bars, ax=ax1, hue=hue, dodge=False, palette=palette)
    ax2 = sns.barplot(x=base, y=bars_pred, ax=ax2, dodge=False, hue=hue_pred, palette=palette)
    change_width(ax1, .55)    
    change_width(ax2, .55)    
    
    # add labels on top of peaks
    plt.sca(ax1)
    for loc in big_ref:
        plt.text(loc, bars[loc]+np.max(bars)/20, loc, ha='center', rotation=90, va='bottom')
    plt.sca(ax2)
    for loc in big:
        if loc > focus[0] and loc < focus[1]:
            plt.text(loc, bars_pred[loc]+np.max(bars_pred)/20, loc, ha='center', rotation=90, va='bottom')
    
    #plot focused
    if log_y:
        ax1.set(yscale="log")
        ax2.set(yscale="log")
        #ax2.sharey(ax1)
        
        ax2.set_ylim(10e-5)
        ax1.sharey(ax2)
    
    ax1.title.set_text(f"Focused reference spectrum of {ref_doc.metadata['name']}")
    ax2.title.set_text(f"Prediction for {len(top_k_ind)+1}. peak")
    ax1.set_xlim(*focus)
    ax2.set_xlim(*focus)
    ax1.xaxis.set_tick_params(labelbottom=True)
    ax2.xaxis.set_tick_params(labelbottom=True)
    ax1.set(ylabel='intensity')
    ax2.set(xlabel='m/z', ylabel='prediction')
    plt.sca(ax1)
    plt.legend(loc='upper right')
    plt.xticks(ticks=np.arange((focus[0]//10)*10, (focus[1]//10)*10, 5 ),fontsize=8, rotation=45)
    plt.sca(ax2)
    plt.legend(loc='upper right')
    plt.xticks(ticks=np.arange((focus[0]//10)*10, (focus[1]//10)*10, 5 ), fontsize=8, rotation=45)
    if save_to_path is not None:
        plt.savefig(save_to_path, bbox_inches='tight')
    plt.show()

def plot_spectrum_predictions_random(ref_doc, omitted_ind, prob, coder, n_detail=10, \
                                     plot_full=True, log_y=False, save_to_path=None):

    matplotlib.rc_file_defaults()

    
    # basic variables
    n_dec = ref_doc.n_decimals
    bars_len = coder.max_mz #len(base)
    base = np.arange(bars_len)
    p_inten = ref_doc.weights
    
    ######################
    # reference spectrum #
    ######################
    
    # create an y axis array of bars
    bars = np.zeros(bars_len)
    for p, peak in enumerate(ref_doc.words):
        #loc = peakname_to_loc(peak, n_dec)
        loc = coder.text_peak_to_mz(peak, n_dec)
        bars[loc] = max(bars[loc],p_inten[p])
      
    
    # distinguish top_k peaks by color
    hue = np.repeat("regular peak", bars_len)
    for ind in omitted_ind:
        hue[ind] = "missing peak"
    
    # distinguish filtered by color
    hue_pred = np.repeat("filtered", bars_len)
    hue_pred[bars == 0] = "returned"
    for ind in omitted_ind:
        hue_pred[ind] = "returned"
    palette ={"filtered": "rosybrown", "returned": "crimson",
              "missing peak": "tab:orange", "regular peak": "tab:blue"}
    
    
    if plot_full:
        # plot it
        f, (ax1, ax2) = plt.subplots(nrows=2, ncols=1,figsize=(40, 10), sharex=True)
        # ax1 = sns.barplot(x=base, y=bars, hue=hue, ax=ax1)
        ax1 = sns.barplot(x=base, y=bars, ax=ax1, dodge=False, hue=hue, palette=palette)
        change_width(ax1, .75)    

    
    # add labels on top of n_details most intensive peaks    
    max_l_ref = np.sort(bars)[-n_detail]
    big_ref = np.nonzero( bars >= max_l_ref)[0]
    
    if plot_full:
        plt.sca(ax1)
        for loc in big_ref:
            plt.text(loc, bars[loc]+np.max(bars)/20, loc, ha='center', rotation=90, va='bottom')

    
    #########################
    # predicted distributon #
    #########################
    
    
    # convert prediction from spec2vec indexing to location on mz indexing
    bars_pred = coder.transform_to_mz(prob, n_dec)
    
    
    # add labels on top of n_details most probable peaks
    max_l = np.sort(bars_pred)[-n_detail]
    big = np.nonzero( bars_pred >= max_l)[0]    
    
    if plot_full:
        plt.sca(ax2)
        for loc in big:
            plt.text(loc, bars_pred[loc]+np.max(bars_pred)/20, loc, ha='center', rotation=90, va='bottom')

        # plot it
        #ax2 = sns.barplot(x=base, y=bars_pred, ax=ax2, color="red")    
        ax2 = sns.barplot(x=base, y=bars_pred, ax=ax2, dodge=False, hue=hue_pred, palette=palette)
        change_width(ax2, .75)    

        
        if log_y:
            ax1.set(yscale="log")
            ax2.set(yscale="log")
            
            ax2.set_ylim(10e-5)
            ax1.sharey(ax2)
        
        ax1.title.set_text(f"Full reference spectrum of {ref_doc.metadata['name']}")
        ax2.title.set_text(f"Predicted distribution for {len(omitted_ind)} peak")
        ax1.xaxis.set_tick_params(labelbottom=True)
        ax1.set(ylabel='intensity')
        ax2.set(xlabel='m/z', ylabel='prediction')
        plt.sca(ax1)
        plt.xticks(ticks=np.arange(0,bars_len,10),fontsize=8, rotation=45)
        plt.legend(loc='upper right')
        plt.sca(ax2)
        plt.legend(loc='upper right')
        plt.xticks(ticks=np.arange(0,bars_len,10), fontsize=8, rotation=45)
        plt.show()

    ##################################
    # focused wiew on the dense area #
    ##################################
    
    # get focus area with np.quantile and nasty hack
    meta = []
    for peak, intensity in zip(ref_doc.words, p_inten):
        meta += [coder.text_peak_to_mz(peak, n_dec)]*int(intensity*1000)
    focus = max(0, np.quantile(meta, 0.05) -50), np.quantile(meta, 0.95) + 50
    
    
    #plot focused 
    f, (ax1, ax2) = plt.subplots(nrows=2, ncols=1,figsize=(20, 10), sharex=True)
    ax1 = sns.barplot(x=base, y=bars, ax=ax1, hue=hue, dodge=False, palette=palette)
    ax2 = sns.barplot(x=base, y=bars_pred, ax=ax2, dodge=False, hue=hue_pred, palette=palette)
    change_width(ax1, .55)    
    change_width(ax2, .55)    

    # add labels on top of peaks
    plt.sca(ax1)
    for loc in omitted_ind:
        plt.text(loc, bars[loc]+np.max(bars)/20, loc, ha='center', rotation=90, va='bottom')
    plt.sca(ax2)
    for loc in big:
        if loc > focus[0] and loc < focus[1]:
            plt.text(loc, bars_pred[loc]+np.max(bars_pred)/20, loc, ha='center', rotation=90, va='bottom')
    
    #plot focused
    if log_y:
        ax1.set(yscale="log")
        ax2.set(yscale="log")
        #ax2.sharey(ax1)
        
        ax2.set_ylim(10e-5)
        ax1.sharey(ax2)
    
    ax1.title.set_text(f"Focused reference spectrum of {ref_doc.metadata['name']}")
    ax2.title.set_text(f"Prediction for {len(omitted_ind)} omitted peaks")
    ax1.set_xlim(*focus)
    ax2.set_xlim(*focus)
    ax1.xaxis.set_tick_params(labelbottom=True)
    ax2.xaxis.set_tick_params(labelbottom=True)
    ax1.set(ylabel='intensity')
    ax2.set(xlabel='m/z', ylabel='prediction')
    plt.sca(ax1)
    plt.legend(loc='upper right')
    plt.xticks(ticks=np.arange((focus[0]//10)*10, (focus[1]//10)*10, 5 ),fontsize=8, rotation=45)
    plt.sca(ax2)
    plt.legend(loc='upper right')
    plt.xticks(ticks=np.arange((focus[0]//10)*10, (focus[1]//10)*10, 5 ), fontsize=8, rotation=45)
    if save_to_path is not None:
        plt.savefig(save_to_path, bbox_inches='tight')

    plt.show()

    
    
    
    
    
    
    
    
    
    
# def peakindex_to_loc(i, n_dec, index2entity):
#     return peakname_to_loc(index2entity[i], n_dec)

# def peakname_to_loc(peakname, n_dec):
#     return int(float(peakname.split("@")[-1])*(10**n_dec)) 

# def pred_to_bars(pred, bars_len, n_dec, index2entity):
#     bars = np.zeros(bars_len)
#     for i in range(len(pred)):
#         loc = peakindex_to_loc(i, n_dec, index2entity)
#         bars[loc] = pred[i]
#     return bars

# def plot_spectrum(ds, model, index=0, n_detail=10, index2entity=None, plot_full=True, log_y=False, k=5):
#     """
#     Parameters
#     ----------
#         ds - torch dataset from which to take the samples
#         model - trained nn.Module model
#         n_detail - number of most intensive peaks with detailed m/z location 
#         index - index of sample in the dataset ds
#     """
    
    
    
#     #ds.onehot = isinstance(model, PureLSTM) #to refactor
#     ds.add_intensity = model.add_intens # TO refactor
# #    index2entity = model.w2v.wv.index2entity
    
#     # TODO add decimals plot support with meaningful plots
    
#     # basic variables
#     ref_doc = ds.ref_docs[index]    
#     n_dec = ref_doc.n_decimals
#     base = np.arange(1001*(10**n_dec))
#     bars_len = len(base)
#     p_inten = ref_doc.weights
    
#     ######################
#     # reference spectrum #
#     ######################
    
#     # create an y axis array of bars
#     bars = np.zeros(len(base))
#     for p, peak in enumerate(ref_doc.words):
#         loc = peakname_to_loc(peak, n_dec)
#         bars[loc] = max(bars[loc],p_inten[p])
      
#     # get top_k_ind
#     top_k_ind, top_after = get_top_k_ind(ref_doc.peaks.intensities, k)
    
#     # distinguish top_k peaks by color
#     hue = np.repeat("regular peak", bars_len)
#     for ind in top_k_ind:
#         hue[peakname_to_loc(ref_doc.words[ind], n_dec)] = f"one of top {len(top_k_ind)}"
    
#     if plot_full:
#         # plot it
#         f, (ax1, ax2) = plt.subplots(nrows=2, ncols=1,figsize=(40, 10), sharex=True)
#         ax1 = sns.barplot(x=base, y=bars, hue=hue, ax=ax1)

    
#     # add labels on top of n_details most intensive peaks    
#     max_l_ref = np.sort(bars)[-n_detail]
#     big_ref = np.nonzero( bars >= max_l_ref)[0]
    
#     if plot_full:
#         plt.sca(ax1)
#         for loc in big_ref:
#             plt.text(loc, bars[loc]+np.max(bars)/20, loc, ha='center', rotation=90, va='bottom')

    
#     #########################
#     # predicted distributon #
#     #########################
    
#     # get top k as model input
#     X, y = ds[index]
    
    
#     #model = model.to("cpu")
    
#     # predict top k+1
#     #with torch.no_grad():
#     #    log_prob, _ = model(X.reshape(1, *X.shape), return_sequence=False)
#     #    prob = np.exp(log_prob)[0]
    
#     prob = model(X.reshape(1, *X.shape))[0]
#     print(prob.shape)
    
#     # convert prediction from spec2vec indexing to location on mz indexing
#     bars_pred = pred_to_bars(prob, bars_len, n_dec, index2entity)
    
    
#     # add labels on top of n_details most probable peaks
#     max_l = np.sort(bars_pred)[-n_detail]
#     big = np.nonzero( bars_pred >= max_l)[0]    
    
#     if plot_full:
#         plt.sca(ax2)
#         for loc in big:
#             plt.text(loc, bars_pred[loc]+np.max(bars_pred)/20, loc, ha='center', rotation=90, va='bottom')

#         # plot it
#         ax2 = sns.barplot(x=base, y=bars_pred, ax=ax2, color="red")    
        
#         if log_y:
#             ax1.set(yscale="log")
#             ax2.set(yscale="log")
            
#             ax2.set_ylim(10e-5)
#             ax1.sharey(ax2)
        
#         ax1.title.set_text(f"Full reference spectrum of {ref_doc.metadata['name']}")
#         ax2.title.set_text(f"Predicted distribution for {len(top_k_ind)+1}. peak")
#         ax1.xaxis.set_tick_params(labelbottom=True)
#         ax1.set(ylabel='intensity')
#         ax2.set(xlabel='m/z', ylabel='probability')
#         plt.sca(ax1)
#         plt.xticks(ticks=np.arange(0,1001,10),fontsize=8, rotation=45)
#         plt.sca(ax2)
#         plt.xticks(ticks=np.arange(0,1001,10), fontsize=8, rotation=45)
#         plt.show()

#     ##################################
#     # focused wiew on the dense area #
#     ##################################
    
#     # get focus area with np.quantile and nasty hack
#     meta = []
#     for peak, intensity in zip(ref_doc.words, p_inten):
#         meta += [peakname_to_loc(peak, n_dec)]*int(intensity*1000)
#     focus = max(0, np.quantile(meta, 0.05) -50), np.quantile(meta, 0.95) + 50
    
    
#     #plot focused 
#     f, (ax1, ax2) = plt.subplots(nrows=2, ncols=1,figsize=(20, 10), sharex=True)
#     ax1 = sns.barplot(x=base, y=bars, ax=ax1, hue=hue)
#     ax2 = sns.barplot(x=base, y=bars_pred, ax=ax2, color="red")
    
#     # add labels on top of peaks
#     plt.sca(ax1)
#     for loc in big_ref:
#         plt.text(loc, bars[loc]+np.max(bars)/20, loc, ha='center', rotation=90, va='bottom')
#     plt.sca(ax2)
#     for loc in big:
#         if loc > focus[0] and loc < focus[1]:
#             plt.text(loc, bars_pred[loc]+np.max(bars_pred)/20, loc, ha='center', rotation=90, va='bottom')
    
#     #plot focused
#     if log_y:
#         ax1.set(yscale="log")
#         ax2.set(yscale="log")
#         #ax2.sharey(ax1)
        
#         ax2.set_ylim(10e-5)
#         ax1.sharey(ax2)
    
#     ax1.title.set_text(f"Focused reference spectrum of {ref_doc.metadata['name']}")
#     ax2.title.set_text(f"Predicted distribution for {len(top_k_ind)+1}. peak")
#     ax1.set_xlim(*focus)
#     ax2.set_xlim(*focus)
#     ax1.xaxis.set_tick_params(labelbottom=True)
#     ax2.xaxis.set_tick_params(labelbottom=True)
#     ax1.set(ylabel='intensity')
#     ax2.set(xlabel='m/z', ylabel='probability')
#     plt.sca(ax1)
#     plt.xticks(ticks=np.arange((focus[0]//10)*10, (focus[1]//10)*10, 5 ),fontsize=8, rotation=45)
#     plt.sca(ax2)
#     plt.xticks(ticks=np.arange((focus[0]//10)*10, (focus[1]//10)*10, 5 ), fontsize=8, rotation=45)
#     plt.show()