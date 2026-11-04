"""Generate a rotatable 3D loss surface with projected contour lines.

Run from this directory with:
    python plot_loss_surface_3d.py
"""

import argparse
from pathlib import Path

import matplotlib.tri as mtri
import numpy as np
import plotly.graph_objects as go

from loss_over_weight_grid import (
    CONSTRAINED_WEIGHTS,
    UNCONSTRAINED_WEIGHTS,
    WEIGHT_NAMES,
    evaluate_losses,
    make_adaptive_surfaces,
    make_axis_grid,
    make_data,
)


def parse_pair(value):
    try:
        pair = tuple(int(index) for index in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("pair must have the form I,J") from error
    if len(pair) != 2 or pair[0] == pair[1] or any(index not in range(7) for index in pair):
        raise argparse.ArgumentTypeError(
            "pair must contain two different indices from 0 to 6"
        )
    return tuple(sorted(pair))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair",
        type=parse_pair,
        default=(0, 1),
        help="Weight-index pair to plot (default: 0,1)",
    )
    parser.add_argument(
        "--context",
        choices=("constrained", "unconstrained", "midpoint"),
        default="constrained",
        help="Fixed values for all other weights (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default="loss_weight_landscapes/loss_surface_3d.html",
        help="Output HTML path (default: %(default)s)",
    )
    parser.add_argument("--sample-size", type=int, default=3072)
    parser.add_argument("--grid-size", type=int, default=21)
    parser.add_argument("--max-grid-size", type=int, default=55)
    parser.add_argument("--refinement-tolerance", type=float, default=0.05)
    parser.add_argument("--range-factor", type=float, default=1.0)
    parser.add_argument("--display-grid", type=int, default=220)
    args = parser.parse_args()
    if not 1 <= args.sample_size <= 10_000:
        parser.error("--sample-size must be between 1 and 10000")
    if args.grid_size < 3 or args.max_grid_size < args.grid_size:
        parser.error("--grid-size must be at least 3 and no larger than --max-grid-size")
    if args.refinement_tolerance <= 0 or args.range_factor <= 0:
        parser.error("--refinement-tolerance and --range-factor must be positive")
    if args.display_grid < 20:
        parser.error("--display-grid must be at least 20")
    return args


def main():
    args = parse_args()
    if args.context == "constrained":
        background = CONSTRAINED_WEIGHTS.copy()
    elif args.context == "unconstrained":
        background = UNCONSTRAINED_WEIGHTS.copy()
    else:
        background = (CONSTRAINED_WEIGHTS + UNCONSTRAINED_WEIGHTS) / 2.0

    i, j = args.pair
    x_grid = make_axis_grid(
        CONSTRAINED_WEIGHTS[i],
        UNCONSTRAINED_WEIGHTS[i],
        args.grid_size,
        args.range_factor,
        0.0,
        background[i],
    )
    y_grid = make_axis_grid(
        CONSTRAINED_WEIGHTS[j],
        UNCONSTRAINED_WEIGHTS[j],
        args.grid_size,
        args.range_factor,
        0.0,
        background[j],
    )
    data = make_data(args.sample_size)
    datasets = (
        (data["training"][0], data["training"][1]),
        (data["extended"][0], data["extended"][1]),
    )
    triangulation, train_losses, extended_losses = make_adaptive_surfaces(
        background,
        args.pair,
        x_grid,
        y_grid,
        datasets,
        64,
        args.max_grid_size,
        args.refinement_tolerance,
    )

    display_x = np.linspace(x_grid[0], x_grid[-1], args.display_grid)
    display_y = np.linspace(y_grid[0], y_grid[-1], args.display_grid)
    display_x_mesh, display_y_mesh = np.meshgrid(display_x, display_y)
    display_grids = []
    for losses in (train_losses, extended_losses):
        log_losses = np.log10(np.maximum(losses, 1e-14))
        interpolator = mtri.LinearTriInterpolator(triangulation, log_losses)
        display_grid = interpolator(display_x_mesh, display_y_mesh)
        display_grids.append(np.asarray(display_grid.filled(np.nan)))

    titles = ("Training data", "Extended data (x scaled by 3)")
    fig = go.Figure()
    finite_surfaces = [surface[np.isfinite(surface)] for surface in display_grids]
    color_min = min(float(np.min(surface)) for surface in finite_surfaces)
    color_max = max(float(np.max(surface)) for surface in finite_surfaces)
    for index, (surface, title) in enumerate(zip(display_grids, titles)):
        fig.add_trace(
            go.Surface(
                x=display_x,
                y=display_y,
                z=surface,
                name=title,
                visible=index == 0,
                colorscale="Viridis",
                cmin=color_min,
                cmax=color_max,
                colorbar=dict(title="log10(MSE)"),
                contours=dict(
                    z=dict(
                        show=True,
                        usecolormap=True,
                        project=dict(z=True),
                        highlightcolor="#ffffff",
                        highlightwidth=2,
                    )
                ),
                hovertemplate=(
                    "{}<br>{}: %{{x:.5g}}<br>{}: %{{y:.5g}}"
                    "<br>log10(MSE): %{{z:.5g}}<extra></extra>"
                ).format(title, WEIGHT_NAMES[i], WEIGHT_NAMES[j]),
            )
        )

    reference_weights = (CONSTRAINED_WEIGHTS, UNCONSTRAINED_WEIGHTS)
    reference_names = ("Constrained pair values", "Unconstrained pair values")
    for data_index, (dataset, title) in enumerate(zip(datasets, titles)):
        reference_losses = []
        for values in reference_weights:
            point = background.copy()
            point[i], point[j] = values[i], values[j]
            reference_losses.append(
                np.log10(
                    max(
                        float(
                            evaluate_losses(
                                point[np.newaxis, :], dataset[0], dataset[1], 64
                            )[0]
                        ),
                        1e-14,
                    )
                )
            )
        fig.add_trace(
            go.Scatter3d(
                x=[values[i] for values in reference_weights],
                y=[values[j] for values in reference_weights],
                z=reference_losses,
                mode="markers",
                name="Reference pair values",
                visible=data_index == 0,
                showlegend=data_index == 0,
                marker=dict(
                    size=7, color=("#ff365e", "#36d9ff"), symbol="diamond"
                ),
                text=reference_names,
                hovertemplate=(
                    "%{{text}}<br>{}: %{{x:.5g}}<br>{}: %{{y:.5g}}"
                    "<br>log10(MSE): %{{z:.5g}}<extra></extra>"
                ).format(WEIGHT_NAMES[i], WEIGHT_NAMES[j]),
            )
        )

    fig.update_layout(
        title=dict(
            text=(
                "Loss landscape: {} vs {}<br>"
                "<sup>{} background; rotate with the mouse, scroll to zoom</sup>"
            ).format(WEIGHT_NAMES[i], WEIGHT_NAMES[j], args.context),
            x=0.5,
        ),
        template="plotly_dark",
        paper_bgcolor="#050505",
        scene=dict(
            bgcolor="#050505",
            xaxis_title=WEIGHT_NAMES[i],
            yaxis_title=WEIGHT_NAMES[j],
            zaxis_title="log10(MSE)",
            dragmode="orbit",
            camera=dict(eye=dict(x=1.65, y=1.65, z=1.2)),
        ),
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.0,
                y=1.08,
                buttons=[
                    dict(
                        label=title,
                        method="update",
                        args=[
                            {
                                "visible": (
                                    [trace_index == 0, trace_index == 1]
                                    + [trace_index == 0, trace_index == 1] * 2
                                )
                            },
                            {"scene.zaxis.title": "log10(MSE)"},
                        ],
                    )
                    for trace_index, title in enumerate(titles)
                ],
            )
        ],
        margin=dict(l=0, r=0, t=90, b=0),
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path, include_plotlyjs=True, full_html=True)
    print(
        "Wrote interactive loss surface to {} ({} adaptive samples).".format(
            output_path.resolve(), len(triangulation.x)
        )
    )


if __name__ == "__main__":
    main()
