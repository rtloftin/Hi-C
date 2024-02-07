
import matplotlib.pyplot as plot
import matplotlib.patches as patches
import matplotlib.cm as colors
import numpy as np


def set_range(min_val, max_val):
    """Sets ranges for an axis that yield for easy-to-read plots."""

    if min_val > max_val:  # No data, set an arbitrary range
        min_val = 0.0
        max_val = 100.0
    elif 0.0 == min_val and 0.0 == max_val:  # All data is zero, set and arbitrary range
        min_val = -100.0
        max_val = 100.0
    elif min_val >= 0.0:  # All values positive, set range from 0 to 120% of max
        min_val = 0.0
        max_val *= 1.2
    elif max_val <= 0.0:  # All values negative, set range from 120% of min to 0
        min_val *= 1.2
        max_val = 0.0
    else:  # Both positive and negative values, expand range by 20%
        min_val *= 1.2
        max_val *= 1.2

    return min_val, max_val


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
        else:
            upper = means
            lower = means

        # Plot series
        plot.plot(x_axis, means, color=color_map[2 * index], alpha=1.0)
        plot.fill_between(x_axis, lower, upper, color=color_map[2 * index + 1], alpha=0.3)

        # Add a legend entry
        legend_entries.append(patches.Patch(color=color_map[2 * index], label=label))

    # Set ranges for the y-axis
    y_min, y_max = set_range(y_min, y_max)

    # Create plot
    plot.legend(handles=legend_entries)
    plot.title(title)
    plot.xlabel(x_label)
    plot.ylabel(y_label)
    plot.ylim(bottom=y_min, top=y_max)
    plot.savefig(image_path, bbox_inches="tight")


def curve_plot(x_values,
               y_values,
               title,
               x_label,
               y_label,
               image_path):
    if not isinstance(x_values, list):
        x_values = [x_values]
    if not isinstance(y_values, list):
        y_values = [y_values]

    x_values = np.asarray(x_values, dtype=np.float32).mean(axis=0)
    y_values = np.asarray(y_values, dtype=np.float32).mean(axis=0)

    x_min, x_max = set_range(x_values.min(), x_values.max())
    y_min, y_max = set_range(y_values.min(), y_values.max())

    color_map = colors.get_cmap("tab20").colors

    plot.clf()
    plot.plot(x_values, y_values, color=color_map[0], alpha=1.0)

    plot.title(title)
    plot.xlabel(x_label)
    plot.ylabel(y_label)
    plot.ylim(bottom=x_min, top=x_max)
    plot.ylim(bottom=y_min, top=y_max)
    plot.savefig(image_path, bbox_inches="tight")
