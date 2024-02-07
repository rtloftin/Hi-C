
import matplotlib.pyplot as plot
import matplotlib.patches as patches
import matplotlib.cm as colors
import numpy as np


def line_plot(data,
              x_axis,
              title,
              x_label,
              y_label,
              image_path,
              errors="range"):
    color_map = colors.get_cmap("tab20").colors
    legend_entries = []
    y_min = np.infty
    y_max = -np.infty

    x_axis = np.asarray(x_axis, dtype=np.float32)

    plot.clf()

    for index, (label, runs) in enumerate(data.items()):
        series = np.asarray(runs, dtype=np.float32)
        means = np.mean(series, axis=0)

        # Update ranges for y-axis
        y_min = min(y_min, np.min(series))
        y_max = max(y_max, np.max(series))

        if "range" == errors:
            upper = np.max(series, axis=0)
            lower = np.min(series, axis=0)
        elif "deviation" == errors:
            std = np.std(series, axis=0, ddof=1)
            upper = means + std
            lower = means - std
        else:  # NOTE: Invalid mode string prints no error bars
            upper = means
            lower = means

        # Plot series
        plot.plot(x_axis, means, color=color_map[2 * index], alpha=1.0)
        plot.fill_between(x_axis, lower, upper, color=color_map[2 * index + 1], alpha=0.3)

        # Add a legend entry
        legend_entries.append(patches.Patch(color=color_map[2 * index], label=label))

    # Set ranges for the y-axis
    if y_min > y_max:  # No data, set an arbitrary range
        y_min = 0.0
        y_max = 100.0
    elif 0.0 == y_min and 0.0 == y_max:  # All data is zero, set and arbitrary range
        y_min = -100.0
        y_max = 100.0
    elif y_min >= 0.0:  # All values positive, set range from 0 to 120% of max
        y_min = 0.0
        y_max *= 1.2
    elif y_max <= 0.0:  # All values negative, set range from 120% of min to 0
        y_min *= 1.2
        y_max = 0.0
    else:  # Both positive and negative values, expand range by 20%
        y_min *= 1.2
        y_max *= 1.2

    # Create plot
    plot.legend(handles=legend_entries)
    plot.title(title)
    plot.xlabel(x_label)
    plot.ylabel(y_label)
    plot.ylim(bottom=y_min, top=y_max)  # NOTE: Sets the range of the y-axis
    plot.savefig(image_path, bbox_inches="tight")
