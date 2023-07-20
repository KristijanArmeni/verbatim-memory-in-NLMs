import os
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")


def set_manuscript_style(style=None):

    if style is not None:
        try:
            plt.style.use(style)
        except:
            logging.info(f"Couldn't find {style} matplotlib style, using default...")

    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = ['Segoe UI']

    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    return 0


def filter_and_aggregate(datain, model, model_id, groups, aggregating_metric):
    """
    Parameters:
    ----------
    datain : dataframe
        dataframe which is filtered and aggregated over
    model : string
        string identifying the model in the 'model' column
    model_id: string
        string identifying the id of the model in the 'model_id' column of datain
    groups : list of dicts
        each element is a dict with column name as key and a list of row values as dict value, 
        the first dict should be the variable that's manipulated (e.g. list length),
        the rest are the ones that are held fixed
    """
    # unpack the dictionaries for each variable
    d1, d2, d3, d4 = groups
   
    # define variables for querying below
    var1 = list(d1.keys())[0]  # e.g. .prompt_len
    var2 = list(d2.keys())[0]  # e.g. .list_len
    var3 = list(d3.keys())[0]  # e.g. .context
    var4 = list(d4.keys())[0]  # e.g. .token_positions

    # select the groups based on variable values
    sel = (datain[var1].isin(d1[var1])) & \
          (datain[var2].isin(d2[var2])) & \
          (datain.second_list.isin(["repeat", "permute", "control"])) & \
          (datain.list.isin(["random", "categorized"])) & \
          (datain[var3].isin(d3[var3])) & \
          (datain.model_id==model_id) & \
          (datain.model==model) & \
          (datain.marker.isin([1, 3])) & \
          (datain[var4].isin(d4[var4]))
    
    if sum(sel) == 0:
        logging.info(f"No rows were selected.\nModel id: {model_id}\nmodel: {model}\nCheck selection conditions.")
        print("This is the input data frame:\n", datain.head())

    d = datain.loc[sel].copy()
    
    ## Aggregate
    # average separately across markers per list_len, stimulus id (sentid), model (lstm or gpt2), marker (1 or 3), list (random, categorized) and second list (repeated, permuted or control)
    units = [var1, "stimid", "model", "marker", "list", "second_list"]

    logging.info(f"Manipulated variable == {var1}")    
    logging.info(f"Aggregating function == {aggregating_metric}")
    logging.info(f"Aggregating over tokens separately per these variables: {units}")
    
    # aggregate with .groupby and .agg
    dagg = d.groupby(units).agg({"surp": [aggregating_metric, "std"], "token": list}).reset_index()
    dagg.columns = ['_'.join(col_v) if col_v[-1] != '' else col_v[0] for col_v in dagg.columns.values]
    
    target_colname = "surp_" + aggregating_metric    
    ## Compute metric
    dataout = relative_change_wrapper(df_agg=dagg, 
                                      groups = [{"model": [model]}, 
                                                 d1,  # this is the manipulated variable
                                                {"second_list": ["repeat", "permute", "control"]},
                                                {"list": ["categorized", "random"]}
                                                ],
                                      compared_col=target_colname,
                                      )
    
    # rename some column/row names for plotting
    dataout.list = dataout.list.map({"categorized": "semantic", "random": "arbitrary"})
    dataout.rename(columns={"second_list": "condition"}, inplace=True)
    dataout.condition = dataout.condition.map({"control": "Novel", "repeat": "Repeated", "permute": "Permuted"})
    
    return dataout, dagg


def get_relative_change(x1=None, x2=None, labels1=None, labels2=None):
    
    """
    computes relative change across data in x1 and x2. Sizes of arrays x1 and x2
    should match.
    """
    
    # check that any labels match
    if (labels1 is not None) & (labels2 is not None):
        assert (labels1 == labels2).all()
    
    x_del = ((x2-x1)/(x1+x2))
    x_perc = (x2/x1)*100
    return x_del, x_perc


def relative_change_wrapper(df_agg, groups, compared_col):
    
    # unpack the dicts
    g1, g2, g3, g4 = groups
    
    # define columns for data
    col1 = list(g1.keys())[0]
    col2 = list(g2.keys())[0]
    col3 = list(g3.keys())[0]
    col4 = list(g4.keys())[0]
    
    # apply relative change computation and apply to 
    df_list = []
    for val1 in list(g1.values())[0]:
        for val2 in list(g2.values())[0]:
            for val3 in list(g3.values())[0]:
                for val4 in list(g4.values())[0]:

                    # initialize output dataframe
                    cols = ["x1", "x2", "x_del"]
                    df = pd.DataFrame(columns=cols)
                        
                    # select the correct rows
                    select = (df_agg.loc[:, col1] == val1) & \
                             (df_agg.loc[:, col2] == val2) & \
                             (df_agg.loc[:, col3] == val3) & \
                             (df_agg.loc[:, col4] == val4)
                    
                    tmp = df_agg.loc[select].copy()

                    # get vectors with aggregated surprisal values from first and second list
                    x1=tmp.loc[tmp.marker==1, compared_col].to_numpy()           # average per sentence surprisal on first list
                    x2=tmp.loc[tmp.marker==3, compared_col].to_numpy()           # average per sentence surprisal on second list
                    labels1 = tmp.loc[tmp.marker==1].stimid.to_numpy()  # use sentence id for check
                    labels2 = tmp.loc[tmp.marker==3].stimid.to_numpy()

                    # compute change and populate output dfs
                    x_del, x_perc = get_relative_change(x1=x1, x2=x2, labels1=labels1, labels2=labels2)
                    
                    df["x1"], df["x2"], df["x_del"], df["x_perc"] = x1, x2, x_del, x_perc
                    df[col1], df[col2], df[col3], df[col4] = val1, val2, val3, val4

                    df_list.append(df)
    
    return pd.concat(df_list)


def make_timecourse_plot(datain, x, style, col, col_order, style_order, hue_order, estimator, err_style):
    
    usetex = plt.rcParams["text.usetex"]
    
    # annotate
    x_levels = datain.loc[:, x].unique()          # x group
    col_levels = datain.loc[:, col].unique()      # column grouping
    style_levels = datain.loc[:, style].unique()
    
    # find n rows (observations) for one plotted group
    one_group = (datain[x] == x_levels[-1]) & (datain[style] == style_levels[0]) & (datain[col] == col_levels[0])
    n = len(datain.loc[one_group])  
    
    p = sns.relplot(kind="line", data=datain, x=x, y="surp", style=style, hue=style, col=col, 
                    estimator=estimator, ci=95.0, err_style=err_style, seed=12345,
                    markers=True, style_order=style_order, hue_order=hue_order, col_order=col_order,
                    legend=True, linewidth=0.7)
    
    # get the data hre
    ax = p.axes[0]
    
    # create a list of lists
    rec = [[col_order[k], style_order[i], int(seg[1, 0]), coll[0].get_ydata()[j], seg[0, 1], seg[1, 1]] 
             for k, a in enumerate(ax)
             for i, coll in enumerate(a.containers) 
             for j, seg in enumerate(coll[2][0].get_segments())]
    
    colnames = [col, style, x, "median", "ci_min", "ci_max"]
    stat = pd.DataFrame.from_records(rec, columns=colnames)
    
    p.set_titles(col_template="{col_name} list")
    p.despine(left=True);
    
    return p, ax, stat


def set_linewidths(axis, lw, n_lines, n_points):
    
    total = n_lines*n_points + n_lines # every line has n_points
    to_consider = np.arange(0, total, int(total/n_lines))
    
    for idx in to_consider:
        axis.lines[idx].set_linewidth(lw)


def get_data_lineplot_with_bars(axis):
    
    """
    extract data from a line plot (errbar) with error bars
    
    Parameters:
    axis : AxisSubplot()
    
    Returns:
     : DataFrame()
    """
    
    labels = axis.get_legend_handles_labels()[-1]
    n_groups = len(labels)
    group_size = int(len(axis.lines)/n_groups)
    
    data_dict = {key: {"est": None, "err": None} for key in labels}
    
    tmplist = []
    for n in range(n_groups):
        
        tmp = pd.DataFrame(columns=["est", "ci_max", "ci_min"])
        
        beg_element = n * group_size
        end_element = beg_element + group_size
        
        tmp["est"] = axis.lines[beg_element].get_data()[-1]
        tmp["ci_min"] = [axis.lines[i].get_data()[-1][0] for i in range(beg_element+1, end_element)]
        tmp["ci_max"] = [axis.lines[i].get_data()[-1][1] for i in range(beg_element+1, end_element)]
        tmp["xlabel"] = [xticklabel.get_text() for xticklabel in axis.get_xticklabels()]
        tmp["hue"] = labels[n]
        tmplist.append(tmp)
    
    return pd.concat(tmplist)


def make_point_plot(data_frame, estimator, x, y, hue, style, col,
                   xlabel=None, ylabel=None, suptitle=None, suptitle_fs=18, ylim=(None, None),
                   size_inches=(5, 3), join=True, scale=1, errwidth=None,
                   legend=False, legend_out=False, custom_legend=True, legend_title=None, 
                   hue_order=["Repeated", "Permuted", "Novel"], col_order=["arbitrary", "semantic"]):
    
    g = sns.catplot(data=data_frame, x=x, y=y, hue=hue, col=col, 
                    estimator=estimator, errorbar=('ci', 95.0),
                    kind="point", join=join, dodge=0.2, scale=scale, errwidth=errwidth,
                    linestyles=["solid", "dotted", "dashed"], markers=['o', 's', 'D'],
                    legend=legend, legend_out=legend_out,
                    seed=12345,
                    hue_order=hue_order, col_order=col_order)
    
    ax = g.axes[0]

    # manually set erorbar color
    #set_color_error_bars(axis=ax[0], color="black", n_lines=3, n_points=4)
    #set_color_error_bars(axis=ax[1], color="black", n_lines=3, n_points=4)
    
    #set_marker_colors(axis=ax[0], facecolor=None, edgecolor="darkgray")
    #set_marker_colors(axis=ax[1], facecolor=None, edgecolor="darkgray")
    
    set_linewidths(axis=ax[0], lw=1.5, n_lines=3, n_points=4)
    set_linewidths(axis=ax[1], lw=1.5, n_lines=3, n_points=4)
    
    # set labels
    ax[0].set_ylabel(ylabel)
    ax[1].set_ylabel("")
    ax[0].set_xlabel(xlabel)
    ax[1].set_xlabel(xlabel)
    
    # set ylim if needed
    if ylim is not None:
        ymin, ymax = ylim
        if ymin is None: ymin = ax[0].get_ylim()[0]
        if ymax is None: ymax = ax[0].get_ylim()[1]
        
        ax[0].set(ylim=(ymin, ymax))
    
    blue, orange, green = sns.color_palette("dark")[0:3]
    n_x_groups = len(data_frame[x].unique())
    
    # annotate
    x_levels = data_frame.loc[:, x].unique()
    col_levels = data_frame.loc[:, col].unique()
    hue_levels = data_frame.loc[:, hue].unique()

    # find n rows for one plotted group
    one_group = (data_frame[x]==x_levels[0]) & (data_frame[hue]==hue_levels[0]) & (data_frame[col] == col_levels[0])
    n = len(data_frame.loc[one_group])  
    print("N per group == {}".format(n))
    
    tmp = []
    for i, a in enumerate(ax):
        tmp_df = get_data_lineplot_with_bars(axis=a)
        tmp_df["cond"] = col_order[i].capitalize()
        tmp.append(tmp_df)
        
    ci_df = pd.concat(tmp)
    
    # legend
    # Improve the legend
    linecolors = plt.rcParams["axes.prop_cycle"].by_key()["color"][0:3]
    if custom_legend:
        handles, labels = ax[1].get_legend_handles_labels()
        ax[1].legend(handles[0:3], labels[0:3], fontsize=16, labelcolor=linecolors,
                     markerscale=1.4, handletextpad=1, columnspacing=1, bbox_to_anchor=(1, 1),
                     loc="upper left", ncol=1, frameon=False, facecolor='white', framealpha=0.2, 
                     title=legend_title, title_fontsize=17)

    g.fig.suptitle(f"{suptitle}", fontsize=suptitle_fs)
    g.set_titles(col_template="{col_name} lists")
    g.despine(left=True)
    g.fig.set_size_inches(size_inches[0], size_inches[1])
    
    return g, ax, ci_df


def set_bars_hatches(axes, patterns):
    """
    taken from: https://stackoverflow.com/questions/55826167/matplotlib-assigning-different-hatch-to-bars
    """
    bars = axes.patches

    hatches = []          # list for hatches in the order of the bars
    for h in patterns:    # loop over patterns to create bar-ordered hatches
        for i in range(int(len(bars) / len(patterns))):
            hatches.append(h)
    for bar, hatch in zip(bars, hatches):  # loop over bars and hatches to set hatches in correct order
        bar.set_hatch(hatch)


def make_bar_plot(data_frame, estimator, x, y, hue, col,
                  ylim=None, xlabel=None, ylabel=None, suptitle=None,
                  size_inches=(5, 3), legend=False, legend_out=False, legend_title=None,
                  hue_order=["Repeated", "Permuted", "Novel"], col_order=["arbitrary", "semantic"]):


    g = sns.catplot(data=data_frame, x=x, y=y, hue=hue, col=col, 
                    estimator=estimator, ci=95.0,
                    kind="bar", dodge=0.5, zorder=2, legend=legend, legend_out=legend_out,
                    seed=12345,
                    hue_order=hue_order, col_order=col_order,
                    edgecolor=["white"], ecolor=["tab:gray"], bottom=0, linewidth=1)
    
    ax = g.axes[0]

    # set hatches manually
    hatch_patterns = ['-', '/', '\\']
    set_bars_hatches(axes=ax[0], patterns=hatch_patterns)
    set_bars_hatches(axes=ax[1], patterns=hatch_patterns)
    
    # set labels
    ax[0].set_ylabel(ylabel)
    ax[0].set_xlabel(xlabel)
    
    if len(ax) == 2:
        ax[1].set_xlabel(xlabel)
        ax[1].set_ylabel("")
        
    # set ylim if needed
    if ylim is not None:
        ymin, ymax = ylim
        if ymin is None: ymin = ax[0].get_ylim()[0]
        if ymax is None: ymax = ax[0].get_ylim()[1]
        
        ax[0].set(ylim=(ymin, ymax))
    
    #blue, orange, green = sns.color_palette("dark")[0:3]
    n_x_groups = len(data_frame[x].unique())
    
    # annotate
    x_levels = data_frame.loc[:, x].unique()
    col_levels = data_frame.loc[:, col].unique()
    hue_levels = data_frame.loc[:, hue].unique()
    
    # find n rows for one plotted group
    one_group = (data_frame[x]==x_levels[0]) & (data_frame[hue]==hue_levels[0]) & (data_frame[col] == col_levels[0])
    n = len(data_frame.loc[one_group])  
    print("N per group == {}".format(n))
    
    # find numerical values
    ci = {}
    for k, a in enumerate(ax):
        d = {lab: [] for lab in hue_order}
        ci[col_order[k]] = {"{}-{}".format(lab, x_levels[j]): plot_obj[0].get_ydata().tolist() + [plot_obj[1].get_height()]
                                                           for i, lab in enumerate(hue_order)
                                                           for j, plot_obj in enumerate(zip(a.lines[0+(i*n_x_groups):n_x_groups+(i*n_x_groups)],
                                                                                            a.patches[0+(i*n_x_groups):n_x_groups+(i*n_x_groups)]))}
    
    # convert dict to a "record" list with observation per row                                                      
    rec = [[key1, key.split("-")[0], key.split("-")[1], ci[key1][key][0], ci[key1][key][1], ci[key1][key][2]] for key1 in ci.keys() for key in ci[key1].keys()]
    
    ci_df = pd.DataFrame.from_records(rec, columns=[col, hue, x, "ci_min", "ci_max", "median"])
    
    # legend
    # Improve the legend
    if not legend:
        
        axis_with_legend = 0
        if len(ax) == 2:
            axis_with_legend = 1
            #ax[0].get_legend().remove()
        
        linecolors = plt.rcParams["axes.prop_cycle"].by_key()["color"][0:3]
        handles, labels = ax[axis_with_legend].get_legend_handles_labels()
        ax[axis_with_legend].legend(handles[0:3], labels[0:3], labelcolor=linecolors,
                                    handletextpad=1, columnspacing=1, bbox_to_anchor=(1, 1),
                                    loc="upper left", ncol=1, frameon=False, title=legend_title)

    g.fig.suptitle("{}".format(suptitle))
    g.set_titles(col_template="{col_name} lists of nouns")
    #g.despine(left=True)
    g.fig.set_size_inches(size_inches[0], size_inches[1])
    
    return g, ax, ci_df