"""Animate pairwise loss slices between the constrained and unconstrained fits.

Run from this directory with:
    python loss_over_weight_grid.py

The default run writes three GIFs for each pair of weights to
``loss_weight_landscapes/``: interpolated, constrained-context, and
unconstrained-context slices. The loss is evaluated with a batched NumPy
forward pass so TensorFlow does not need to be imported for the grid sweep.
"""

import argparse
import itertools
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.tri import Triangulation
import numpy as np


UNCONSTRAINED_WEIGHTS = np.array(
    [-1.0866661, 0.85507196, -3.102202, -2.7769487, 51.564045, 91.12118, -6.4819527],
    dtype=np.float64,
)
# The constrained final-weight snapshot in test_quadratic.py has bias 189.08543.
CONSTRAINED_WEIGHTS = np.array(
    [0.18998857, 0.39265335, 0.0, -1.316958, -559.892, 434.2152, 189.08543],
    dtype=np.float64,
)
WEIGHT_NAMES = (
    "hidden_kernel_1",
    "hidden_kernel_2",
    "hidden_bias_1",
    "hidden_bias_2",
    "output_weight_1",
    "output_weight_2",
    "output_bias",
)
NUM_TRAINING_SAMPLES = 10_000


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="loss_weight_landscapes",
        help="Directory for generated GIFs (default: %(default)s)",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=13,
        help="Initial coarse samples per weight axis before adaptive refinement (default: %(default)s)",
    )
    parser.add_argument(
        "--max-grid-size",
        type=int,
        default=33,
        help="Maximum adaptive samples per axis (default: %(default)s)",
    )
    parser.add_argument(
        "--refinement-tolerance",
        type=float,
        default=0.08,
        help="Refine where midpoint log10(MSE) error exceeds this value (default: %(default)s)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=31,
        help="Frames per GIF; must be at least 2 to include both endpoints (default: %(default)s)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=2048,
        help="Deterministic sample size for both data panels; 0 uses all 10000 samples (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Number of grid points evaluated at once (default: %(default)s)",
    )
    parser.add_argument(
        "--range-factor",
        type=float,
        default=1.0,
        help="Extra broad-range margin relative to each weight's scale (default: %(default)s)",
    )

    parser.add_argument(
        "--pairs",
        nargs="+",
        metavar="I,J",
        help="Weight-index pairs to animate (default: all 21 pairs), e.g. --pairs 0,1 4,6",
    )
    args = parser.parse_args()

    if args.grid_size < 9:
        parser.error("--grid-size must be at least 9")
    if args.max_grid_size < args.grid_size:
        parser.error("--max-grid-size must be at least --grid-size")
    if args.refinement_tolerance <= 0:
        parser.error("--refinement-tolerance must be positive")
    if args.frames < 2:
        parser.error("--frames must be at least 2")
    if args.sample_size < 0 or args.sample_size > NUM_TRAINING_SAMPLES:
        parser.error("--sample-size must be 0 (all data) or between 1 and 10000")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.range_factor <= 0:
        parser.error("--range-factor must be positive")

    return args


def get_pairs(pair_args):
    if pair_args is None:
        return list(itertools.combinations(range(len(WEIGHT_NAMES)), 2))

    pairs = []
    for pair_arg in pair_args:
        try:
            pair = tuple(int(part) for part in pair_arg.split(","))
        except ValueError:
            raise ValueError("Each --pairs value must have the form I,J")
        if len(pair) != 2 or pair[0] == pair[1]:
            raise ValueError("Each pair must contain two different weight indices")
        if any(index < 0 or index >= len(WEIGHT_NAMES) for index in pair):
            raise ValueError("Weight indices must be in the range 0..6")
        pair = tuple(sorted(pair))
        if pair not in pairs:
            pairs.append(pair)
    if not pairs:
        raise ValueError("At least one weight pair is required")
    return pairs


def make_data(sample_size):
    # Matches quadratic_model.py's np.random.seed(1); x1 = np.random.randn(10000).
    x = np.random.RandomState(1).randn(NUM_TRAINING_SAMPLES)
    y = 1.0 + 2.0 * x + 3.0 * x**2
    extended_x = 3.0 * x
    extended_y = 1.0 + 2.0 * extended_x + 3.0 * extended_x**2

    if sample_size:
        indices = np.linspace(0, NUM_TRAINING_SAMPLES - 1, sample_size, dtype=np.int64)
        x, y = x[indices], y[indices]
        extended_x, extended_y = extended_x[indices], extended_y[indices]

    sample_label = "{:,} of {:,} samples".format(len(x), NUM_TRAINING_SAMPLES)
    return {
        "training": (x, y, "Training data ({})".format(sample_label)),
        "extended": (
            extended_x,
            extended_y,
            "Extended data, x scaled by 3 ({})".format(sample_label),
        ),
    }


def make_axis_grid(
    constrained_value,
    unconstrained_value,
    point_count,
    range_factor,
    zoom,
    current_value,
):
    """Build a uniform grid that transitions from the broad range to a local view."""
    scale = max(
        abs(constrained_value),
        abs(unconstrained_value),
        abs(unconstrained_value - constrained_value),
        1.0,
    )
    low = min(constrained_value, unconstrained_value) - range_factor * scale
    high = max(constrained_value, unconstrained_value) + range_factor * scale
    local_radius = 0.08 * scale
    local_low = current_value - local_radius
    local_high = current_value + local_radius

    return np.linspace(
        (1.0 - zoom) * low + zoom * local_low,
        (1.0 - zoom) * high + zoom * local_high,
        point_count,
    )


def evaluate_points(background, pair, points, x, y, batch_size):
    i, j = pair
    weight_points = np.broadcast_to(background, (len(points), 7)).copy()
    weight_points[:, i] = points[:, 0]
    weight_points[:, j] = points[:, 1]
    return evaluate_losses(weight_points, x, y, batch_size)


def make_adaptive_surfaces(
    background,
    pair,
    x_grid,
    y_grid,
    datasets,
    batch_size,
    max_grid_size,
    tolerance,
):
    """Locally refine triangles with large midpoint interpolation error."""
    x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
    points = np.column_stack((x_mesh.ravel(), y_mesh.ravel()))
    max_point_count = max_grid_size**2
    while True:
        triangulation = Triangulation(points[:, 0], points[:, 1])
        losses = tuple(
            evaluate_points(background, pair, points, x, y, batch_size)
            for x, y in datasets
        )
        surfaces = (
            np.log10(np.maximum(losses[0], 1e-14)),
            np.log10(np.maximum(losses[1], 1e-14)),
        )
        triangles = triangulation.triangles
        vertices = points[triangles]
        edge_pairs = np.array(((0, 1), (1, 2), (2, 0)))
        edge_lengths = np.sum(
            (vertices[:, edge_pairs[:, 0]] - vertices[:, edge_pairs[:, 1]]) ** 2,
            axis=2,
        )
        longest_edges = edge_pairs[np.argmax(edge_lengths, axis=1)]
        centers = np.mean(vertices, axis=1)
        edge_midpoints = np.mean(
            vertices[np.arange(len(triangles))[:, np.newaxis], longest_edges], axis=1
        )
        candidates = np.concatenate((centers, edge_midpoints), axis=0)

        predictions = []
        for surface in surfaces:
            center_prediction = np.mean(surface[triangles], axis=1)
            edge_prediction = np.mean(
                surface[triangles[np.arange(len(triangles))[:, np.newaxis], longest_edges]],
                axis=1,
            )
            predictions.append(np.concatenate((center_prediction, edge_prediction)))

        candidate_losses = tuple(
            evaluate_points(background, pair, candidates, x, y, batch_size)
            for x, y in datasets
        )
        errors = np.maximum(
            np.abs(np.log10(np.maximum(candidate_losses[0], 1e-14)) - predictions[0]),
            np.abs(np.log10(np.maximum(candidate_losses[1], 1e-14)) - predictions[1]),
        )
        selected = np.flatnonzero(errors > tolerance)
        capacity = max_point_count - len(points)
        if capacity <= 0 or not len(selected):
            return triangulation, losses[0], losses[1]
        selected = selected[np.argsort(errors[selected])[::-1]]
        existing = {tuple(point) for point in np.round(points, decimals=12)}
        additions = []
        for candidate_index in selected:
            candidate = candidates[candidate_index]
            key = tuple(np.round(candidate, decimals=12))
            if key not in existing:
                existing.add(key)
                additions.append(candidate)
                if len(additions) == capacity:
                    break
        if not len(additions):
            return triangulation, losses[0], losses[1]
        points = np.concatenate((points, np.asarray(additions)), axis=0)


def make_frame_schedule(frame_count):
    """Show the global landscape, then zoom at each reference neighborhood."""
    progress = np.linspace(0.0, 1.0, frame_count)
    background_t = np.interp(progress, [0.0, 0.36, 0.70, 1.0], [0.0, 0.0, 1.0, 1.0])
    zoom = np.interp(
        progress, [0.0, 0.18, 0.36, 0.70, 1.0], [0.0, 1.0, 0.0, 0.0, 1.0]
    )
    return list(zip(background_t, zoom))


def sigmoid(values):
    return 1.0 / (1.0 + np.exp(-np.clip(values, -60.0, 60.0)))


def evaluate_losses(weight_points, x, y, batch_size):
    """Evaluate MSE for many flattened [w0, ..., w6] vectors in batches."""
    losses = np.empty(len(weight_points), dtype=np.float64)
    x_row = x[np.newaxis, :]
    y_row = y[np.newaxis, :]

    for start in range(0, len(weight_points), batch_size):
        stop = min(start + batch_size, len(weight_points))
        weights = weight_points[start:stop]

        hidden_1 = sigmoid(weights[:, 0, np.newaxis] * x_row + weights[:, 2, np.newaxis])
        hidden_2 = sigmoid(weights[:, 1, np.newaxis] * x_row + weights[:, 3, np.newaxis])
        predictions = (
            hidden_1 * weights[:, 4, np.newaxis]
            + hidden_2 * weights[:, 5, np.newaxis]
            + weights[:, 6, np.newaxis]
        )
        residuals = predictions - y_row
        losses[start:stop] = np.mean(residuals * residuals, axis=1)

    return losses


def format_duration(seconds):
    if not np.isfinite(seconds):
        return "calculating"
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return "{}h {:02d}m {:02d}s".format(hours, minutes, seconds)
    if minutes:
        return "{}m {:02d}s".format(minutes, seconds)
    return "{}s".format(seconds)


class Progress:
    def __init__(self, total):
        self.total = total
        self.completed = 0
        self.started = time.monotonic()
        self.print_interval = max(1, total // 40)
        print("Progress:   0.0% | ETA calculating", flush=True)

    def advance(self, task):
        self.completed += 1
        if self.completed % self.print_interval != 0 and self.completed != self.total:
            return
        elapsed = time.monotonic() - self.started
        eta = elapsed * (self.total - self.completed) / self.completed
        percentage = 100.0 * self.completed / self.total
        print(
            "Progress: {:5.1f}% ({}/{}) | ETA {} | {}".format(
                percentage,
                self.completed,
                self.total,
                format_duration(eta),
                task,
            ),
            flush=True,
        )


def render_animation(
    path,
    train_losses_by_frame,
    validation_losses_by_frame,
    frame_states,
    grids_by_frame,
    pair,
    train_label,
    validation_label,
    constrained_weights,
    unconstrained_weights,
    context_name,
    progress,
):
    train_log_surfaces = [
        np.log10(np.maximum(losses, 1e-14)) for losses in train_losses_by_frame
    ]
    validation_log_surfaces = [
        np.log10(np.maximum(losses, 1e-14)) for losses in validation_losses_by_frame
    ]
    all_surfaces = train_log_surfaces + validation_log_surfaces
    vmin = min(float(np.min(surface)) for surface in all_surfaces)
    vmax = max(float(np.max(surface)) for surface in all_surfaces)
    if vmax <= vmin:
        vmax = vmin + 1.0
    norm = Normalize(vmin=vmin, vmax=vmax)
    levels = np.linspace(vmin, vmax, 32)

    fig, axes = plt.subplots(
        1, 2, figsize=(15, 7), sharex=True, sharey=True, constrained_layout=True
    )
    mappable = ScalarMappable(norm=norm, cmap="viridis")
    mappable.set_array([])
    fig.colorbar(
        mappable,
        ax=axes,
        label="log10(MSE), shared across both panels and all frames",
    )

    i, j = pair
    fig.suptitle(
        "{} vs {} — {} background\n"
        "Broad landscape first; zooms into constrained and unconstrained neighborhoods.".format(
            WEIGHT_NAMES[i], WEIGHT_NAMES[j], context_name
        ),
        fontsize=11,
    )

    fps = max(1, min(8, len(train_losses_by_frame) // 5))
    writer = PillowWriter(fps=fps)
    with writer.saving(fig, str(path), dpi=90):
        frame_count = len(train_losses_by_frame)
        for frame_index, (state, (triangulation, bounds)) in enumerate(
            zip(frame_states, grids_by_frame)
        ):
            x_low, x_high, y_low, y_high = bounds
            train_log = train_log_surfaces[frame_index]
            validation_log = validation_log_surfaces[frame_index]
            for ax, log_surface, panel_label, current_mse in zip(
                axes,
                (train_log, validation_log),
                (train_label, validation_label),
                (state[3], state[4]),
            ):
                ax.clear()
                ax.tricontourf(
                    triangulation,
                    log_surface,
                    levels=levels,
                    cmap="viridis",
                    norm=norm,
                    extend="both",
                )
                ax.scatter(
                    constrained_weights[i],
                    constrained_weights[j],
                    marker="*",
                    s=150,
                    color="crimson",
                    edgecolors="white",
                    linewidths=0.7,
                    label="Constrained reference (start)",
                    zorder=4,
                )
                ax.scatter(
                    unconstrained_weights[i],
                    unconstrained_weights[j],
                    marker="*",
                    s=150,
                    color="deepskyblue",
                    edgecolors="black",
                    linewidths=0.5,
                    label="Unconstrained solution (end)",
                    zorder=4,
                )
                ax.scatter(
                    state[0],
                    state[1],
                    marker="o",
                    s=55,
                    color="black",
                    edgecolors="white",
                    linewidths=0.8,
                    label="Current background state",
                    zorder=5,
                )
                ax.set_xlim(x_low, x_high)
                ax.set_ylim(y_low, y_high)
                ax.set_xlabel("{} (weight {})".format(WEIGHT_NAMES[i], i))
                ax.set_title(
                    "{} (t={:.3f})\nCurrent MSE={:.6g}".format(
                        panel_label, state[2], current_mse
                    ),
                    fontsize=9,
                )
                ax.legend(loc="best", fontsize=7, framealpha=0.9)

            axes[0].set_ylabel("{} (weight {})".format(WEIGHT_NAMES[j], j))
            writer.grab_frame()
            progress.advance(
                "saving {} / frame {}/{}".format(path.name, frame_index + 1, frame_count)
            )

    plt.close(fig)


def main():
    args = parse_args()
    try:
        pairs = get_pairs(args.pairs)
    except ValueError as error:
        raise SystemExit(str(error))

    datasets = make_data(args.sample_size)
    train_x, train_y, train_label = datasets["training"]
    validation_x, validation_y, validation_data_label = datasets["extended"]
    validation_label = "Validation/test: {}".format(validation_data_label)
    frame_schedule = make_frame_schedule(args.frames)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    contexts = (
        ("interpolated", None),
        ("constrained_context", CONSTRAINED_WEIGHTS),
        ("unconstrained_context", UNCONSTRAINED_WEIGHTS),
    )
    total_tasks = len(pairs) * len(contexts) * args.frames * 2
    progress = Progress(total_tasks)
    sample_count = len(train_x)
    print(
        "Evaluating {} pair(s) across {} background paths ({} animations), {} frame(s) each, {} sample(s) in each panel.".format(
            len(pairs),
            len(contexts),
            len(pairs) * len(contexts),
            args.frames,
            sample_count,
        ),
        flush=True,
    )

    for i, j in pairs:
        for context_name, fixed_background in contexts:
            train_losses_by_frame = []
            validation_losses_by_frame = []
            frame_states = []
            grids_by_frame = []

            for frame_index, (t, zoom) in enumerate(frame_schedule):
                current_weights = (
                    (1.0 - t) * CONSTRAINED_WEIGHTS
                    + t * UNCONSTRAINED_WEIGHTS
                )
                background = current_weights.copy()
                if fixed_background is not None:
                    background[:] = fixed_background
                    background[i] = current_weights[i]
                    background[j] = current_weights[j]
                current_state_weights = background.copy()

                x_grid = make_axis_grid(
                    CONSTRAINED_WEIGHTS[i],
                    UNCONSTRAINED_WEIGHTS[i],
                    args.grid_size,
                    args.range_factor,
                    zoom,
                    current_weights[i],
                )
                y_grid = make_axis_grid(
                    CONSTRAINED_WEIGHTS[j],
                    UNCONSTRAINED_WEIGHTS[j],
                    args.grid_size,
                    args.range_factor,
                    zoom,
                    current_weights[j],
                )
                triangulation, train_losses, validation_losses = (
                    make_adaptive_surfaces(
                        background,
                        (i, j),
                        x_grid,
                        y_grid,
                        ((train_x, train_y), (validation_x, validation_y)),
                        args.batch_size,
                        args.max_grid_size,
                        args.refinement_tolerance,
                    )
                )
                train_losses_by_frame.append(train_losses)
                validation_losses_by_frame.append(validation_losses)
                grids_by_frame.append(
                    (
                        triangulation,
                        (x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]),
                    )
                )

                current_train_loss = float(
                    evaluate_losses(
                        current_state_weights[np.newaxis, :],
                        train_x,
                        train_y,
                        args.batch_size,
                    )[0]
                )
                current_validation_loss = float(
                    evaluate_losses(
                        current_state_weights[np.newaxis, :],
                        validation_x,
                        validation_y,
                        args.batch_size,
                    )[0]
                )
                frame_states.append(
                    (
                        current_weights[i],
                        current_weights[j],
                        t,
                        current_train_loss,
                        current_validation_loss,
                    )
                )
                progress.advance(
                    "evaluating {} {} vs {} / frame {}/{}".format(
                        context_name,
                        WEIGHT_NAMES[i],
                        WEIGHT_NAMES[j],
                        frame_index + 1,
                        args.frames,
                    )
                )

            suffix = "" if context_name == "interpolated" else "_{}".format(context_name)
            filename = "loss_{}_vs_{}_train_vs_validation{}.gif".format(
                WEIGHT_NAMES[i], WEIGHT_NAMES[j], suffix
            )
            render_animation(
                output_dir / filename,
                train_losses_by_frame,
                validation_losses_by_frame,
                frame_states,
                grids_by_frame,
                (i, j),
                train_label,
                validation_label,
                CONSTRAINED_WEIGHTS,
                UNCONSTRAINED_WEIGHTS,
                context_name.replace("_", " "),
                progress,
            )

    elapsed = time.monotonic() - progress.started
    print(
        "Completed {} side-by-side animation(s) in {}. Output: {}".format(
            len(pairs), format_duration(elapsed), output_dir.resolve()
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
