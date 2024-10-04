import sys, os
import itertools
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Load the correct matplotlibrc
mpl.rc_file(os.path.join(os.path.dirname(__file__), 'matplotlibrc'))
mpl.rcParams["text.usetex"] = True
mpl.rcParams["text.latex.preamble"] = r"""
    \usepackage{siunitx}
    \usepackage{libertine}
    \usepackage{libertinust1math}
    \usepackage{mathrsfs}
    \usepackage{amssymb}
    \renewcommand*\familydefault{\sfdefault}
    \renewcommand{\vec}[1]{\mathbf{#1}}
    \newcommand{\mat}[1]{\mathbf{#1}}
"""

blues = ["#729fcfff", "#3465a4ff", "#204a87ff"][::-1]
reds = ["#ef2929ff", "#cc0000ff", "#a40000ff"][::-1]
greens = ["#8ae234ff", "#73d216ff", "#4e9a06ff"][::-1]
oranges = ["#fcaf3eff", "#f57900ff", "#ce5c00ff"][::-1]
purples = ["#ad7fa8ff", "#75507bff", "#5c3566ff"][::-1]
yellows = ["#fce94fff", "#edd400ff", "#c4a000ff"][::-1]
grays = [
    "#eeeeecff", "#d3d7cfff", "#babdb6ff", "#888a85ff", "#555753ff",
    "#2e3436ff"
][::-1]
cols = list(itertools.chain.from_iterable(zip(blues,reds,greens,oranges,purples,yellows)))

def save(fig, filename, suffix=""):
    # Special treatment for PDFs. We need to run the resulting PDF
    # through Ghostscript to
    # a) trim the figures properly
    # b) subset fonts
    target_file, target_ext = os.path.splitext(filename)
    target_file += suffix
    if target_ext == ".pdf":
        target = target_large = target_file + ".large" + target_ext
    else:
        target = target_file + target_ext
    print("Saving to", target)
    fig.savefig(target,
                bbox_inches='tight',
                pad_inches=0.05,
                transparent=True)

    if target_ext == ".pdf":
        import subprocess
        import re

        # Extract the bounding box
        print("Extracting bounding box of file", target_large)
        gs_out = subprocess.check_output(
            ["gs", "-o", "-", "-sDEVICE=bbox", target_large],
            stderr=subprocess.STDOUT)
        pattern = r"^%%HiResBoundingBox:\s*([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*$"
        x0, y0, x1, y1 = map(
            float,
            re.search(pattern, str(gs_out, "utf-8"),
                      re.MULTILINE).groups())

        # Add a small extension to the bounding box
        pad = 0.5
        x0 -= pad
        x1 += pad
        y0 -= pad
        y1 += pad

        # Run ghostscript again to crop the file
        # See https://stackoverflow.com/a/46058965
        target = target_file + target_ext
        print("Optimising PDF and saving to", target)
        subprocess.check_output([
            "gs", "-o", target, "-dEmbedAllFonts=true",
            "-dSubsetFonts=true", "-dCompressFonts=true",
            "-dPDFSETTINGS=/prepress", '-dDoThumbnails=false',
            "-sDEVICE=pdfwrite", "-dCompatibilityLevel=1.5",
            f"-dDEVICEWIDTHPOINTS={x1 - x0}",
            f"-dDEVICEHEIGHTPOINTS={y1 - y0}", "-dFIXEDMEDIA", "-c",
            f"<</PageOffset [-{x0} -{y0}]>>setpagedevice", "-f",
            target_large
        ])

        # Remove the large temporary file
        os.unlink(target_large)

def outside_ticks(ax):
    ax.tick_params(direction="out", which="both")

def remove_frame(ax):
    for spine in ["left", "right", "top", "bottom"]:
        ax.spines[spine].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticks([], minor=True)
    ax.set_yticks([], minor=True)

def add_frame(self, ax):
    for spine in ["left", "right", "top", "bottom"]:
        ax.spines[spine].set_visible(True)

def annotate(ax,
             x0,
             y0,
             x1,
             y1,
             s=None,
             ha="center",
             va="center",
             fontdict=None,
             zorder=None,
             color="k"):
    ax.plot([x0, x1], [y0, y1],
            color=color,
            linewidth=0.5,
            linestyle=(0, (1, 1)),
            clip_on=False,
            zorder=zorder)
    ax.plot([x0], [y0],
            'o',
            color=color,
            markersize=1,
            clip_on=False,
            zorder=zorder)
    if not s is None:
        ax.text(x1,
                y1,
                s,
                ha=ha,
                va=va,
                bbox={
                    "pad": 1.0,
                    "color": "w",
                    "linewidth": 0.0,
                },
                fontdict=fontdict,
                zorder=zorder)


def vslice(ax, x, y0, y1, **kwargs):
    ax.plot([x, x], [y0, y1],
            'k-',
            linewidth=0.75,
            clip_on=False,
            **kwargs)
    ax.plot(x, y0, 'k_', linewidth=0.75, clip_on=False, **kwargs)
    ax.plot(x, y1, 'k_', linewidth=0.75, clip_on=False, **kwargs)


def hslice(ax, x0, x1, y, **kwargs):
    ax.plot([x0, x1], [y, y],
            'k-',
            linewidth=0.75,
            clip_on=False,
            **kwargs)
    ax.plot(x0, y, 'k|', linewidth=0.75, clip_on=False, **kwargs)
    ax.plot(x1, y, 'k|', linewidth=0.75, clip_on=False, **kwargs)


def timeslice(ax, x0, x1, y, **kwargs):
    hslice(ax, x0, x1, y, **kwargs)



def reorderLegend(ax=None,order=None,unique=False,**kwargs):
    if ax is None: ax=plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0])) # sort both labels and handles by labels
    if order is not None: # Sort according to a given list (not necessarily complete)
        keys=dict(zip(order,range(len(order))))
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t,keys=keys: keys.get(t[0],np.inf)))
    if unique:  labels, handles= zip(*unique_everseen(zip(labels,handles), key = labels)) # Keep only the first of each handle
    ax.legend(handles, labels,**kwargs)
    return(handles, labels)


def unique_everseen(seq, key=None):
    seen = set()
    seen_add = seen.add
    return [x for x,k in zip(seq,key) if not (k in seen or seen_add(k))]


import seaborn as sns
from rliable.plot_utils import _annotate_and_decorate_axis
def plot_sample_efficiency_curve(frames,
                                 point_estimates,
                                 interval_estimates,
                                 algorithms=None,
                                 colors=None,
                                 labels=None,
                                 linestys=None,
                                 color_palette='colorblind',
                                 figsize=(7, 5),
                                 xlabel=r'Number of Frames (in millions)',
                                 ylabel='Aggregate Human Normalized Score',
                                 ax=None,
                                 labelsize=9.0,
                                 ticklabelsize=9.0,
                                 marker=',',
                                 **kwargs):
  """Plots an aggregate metric with CIs as a function of environment frames.

  Args:
    frames: Array or list containing environment frames to mark on the x-axis.
    point_estimates: Dictionary mapping algorithm to a list or array of point
      estimates of the metric corresponding to the values in `frames`.
    interval_estimates: Dictionary mapping algorithms to interval estimates
      corresponding to the `point_estimates`. Typically, consists of stratified
      bootstrap CIs.
    algorithms: List of methods used for plotting. If None, defaults to all the
      keys in `point_estimates`.
    colors: Dictionary that maps each algorithm to a color. If None, then this
      mapping is created based on `color_palette`.
    color_palette: `seaborn.color_palette` object for mapping each method to a
      color.
    figsize: Size of the figure passed to `matplotlib.subplots`. Only used when
      `ax` is None.
    xlabel: Label for the x-axis.
    ylabel: Label for the y-axis.
    ax: `matplotlib.axes` object.
    labelsize: Font size of the x-axis label.
    ticklabelsize: Font size of the ticks.
    **kwargs: Arbitrary keyword arguments.

  Returns:
    `axes.Axes` object containing the plot.
  """
  if ax is None:
    _, ax = plt.subplots(figsize=figsize)
  if algorithms is None:
    algorithms = list(point_estimates.keys())
  if colors is None:
    color_palette = sns.color_palette(color_palette, n_colors=len(algorithms))
    colors = dict(zip(algorithms, color_palette))



  if labels is None:
      labels = dict(zip(algorithms,algorithms))
  if labels is None:
      labels = dict(zip(algorithms, ['-'] * len(algorithms)))
      

  for algorithm in algorithms:
    metric_values = point_estimates[algorithm]
    lower, upper = interval_estimates[algorithm]
    ax.plot(
        frames,
        metric_values,
        color=colors[algorithm],
        marker=marker,
        linewidth=1,
        label=labels[algorithm],
        linestyle=linestys[algorithm])

    ax.fill_between(
        frames, y1=lower, y2=upper, color=colors[algorithm], alpha=0.1)
  kwargs.pop('marker', '0')
  kwargs.pop('linewidth', '2')

  return _annotate_and_decorate_axis(
      ax,
      xlabel=xlabel,
      ylabel=ylabel,
      labelsize=labelsize,
      ticklabelsize=ticklabelsize,
      **kwargs)


def make_plots(model_names, env_name, linestys, cols, labels, n_seeds=1,ax=None,
               legend=False,leg_order=None,ma_window=100,
               **kwargs):
    import pandas as pd
    from rliable import library as rly
    from rliable import metrics
    from .storage import get_model_dir
    if ax is None:
        fig,ax =plt.subplots(1, 1, figsize=(7,2))
    
    res_dict = dict(zip(model_names, [[] for i in range(len(model_names))]))
    # frames_dict = {}
    for i,model_name in enumerate(model_names):
        for seed in range(0,n_seeds):
            try:
                model_dir = get_model_dir(model_name +  '_' + str(seed))
                data = pd.read_csv(model_dir + "/log.csv")
            except:
                print(f"Cannot find {model_dir}")
                pass
                data_returns = np.convolve(pd.to_numeric(data['return_mean']).values, np.ones(ma_window)/ma_window, mode='valid')
                res_dict[model_name].append(data_returns)#.rolling(100).mean()
            
        res_dict[model_name] = np.array(res_dict[model_name])[:,None,:]
        # frames_dict[model_name] = pd.to_numeric(data['frames']).astype(float).values
    frames = pd.to_numeric(data['frames']).astype(float).values[:res_dict[model_name].shape[-1]]
    iqm = lambda scores: np.array([metrics.aggregate_iqm(scores[..., frame])
                               for frame in range(scores.shape[-1])])
    iqm_scores, iqm_cis = rly.get_interval_estimates(
          res_dict, iqm, reps=10000)
    for i,model_name in enumerate(model_names):
        lower, upper = iqm_cis[model_name]
        ax.plot(
            frames,
            iqm_scores[model_name],
            color=cols[model_name],
            label=labels[model_name],
            linestyle=linestys[model_name], **kwargs)
        ax.fill_between(
            frames, y1=lower, y2=upper, color=cols[model_name], alpha=0.2)
        
    ax.set_xlabel("Frames observed")
    ax.set_ylabel("Average return")
    
    ax.set_title(env_name)
    ax.ticklabel_format(style='scientific',scilimits=(-1,1))
    if legend:
        ax.legend(frameon=1,edgecolor='white')
        if leg_order is not None:
            reorderLegend(ax,leg_order,frameon=1,edgecolor='white')
    return frames, iqm_scores, iqm_cis, res_dict

