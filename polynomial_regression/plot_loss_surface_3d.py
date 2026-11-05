"""Generate a rotatable 3D loss surface with projected contour lines.

Run from this directory with:
    python plot_loss_surface_3d.py
"""

import argparse
import itertools
from pathlib import Path

import matplotlib.tri as mtri
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from loss_over_weight_grid import (
    CONSTRAINED_WEIGHTS,
    UNCONSTRAINED_WEIGHTS,
    WEIGHT_NAMES,
    evaluate_losses,
    make_adaptive_surfaces,
    make_axis_grid,
    make_data,
)

CAMERA_SYNC_SCRIPT = r"""
(function () {
    const graph = document.getElementById('{plot_id}');
    let activeSourceScene = null;
    let pendingSourceScene = null;
    let frameHandle = null;

    function sceneFromEvent(eventData) {
        const keys = Object.keys(eventData);
        if (keys.some(function (key) {
            return key === 'scene.camera' || key.indexOf('scene.camera.') === 0;
        })) {
            return 'scene';
        }
        if (keys.some(function (key) {
            return key === 'scene2.camera' || key.indexOf('scene2.camera.') === 0;
        })) {
            return 'scene2';
        }
        return null;
    }

    function syncScene(sourceName) {
        const targetName = sourceName === 'scene' ? 'scene2' : 'scene';
        const source = graph._fullLayout[sourceName]._scene;
        const target = graph._fullLayout[targetName]._scene;
        const camera = source.getCamera();
        const matrix = Array.from(source.camera.view.computedMatrix);

        graph.layout[sourceName].camera = camera;
        graph.layout[targetName].camera = JSON.parse(JSON.stringify(camera));
        target.camera.delay = 0;
        target.camera.view.setMatrix(target.camera.view.lastT(), matrix);
        target.camera.view.recalcMatrix(target.camera.view.lastT());
        target.glplot.redraw();
    }

    function flushPendingCamera() {
        frameHandle = null;
        if (!pendingSourceScene) {
            return;
        }
        const sourceName = pendingSourceScene;
        pendingSourceScene = null;
        syncScene(sourceName);
    }

    function synchronizeCamera(eventData, isFinal) {
        const sceneName = sceneFromEvent(eventData);
        if (!sceneName) {
            return;
        }
        activeSourceScene = activeSourceScene || sceneName;
        if (activeSourceScene && sceneName !== activeSourceScene) {
            return;
        }

        if (isFinal) {
            if (frameHandle !== null) {
                cancelAnimationFrame(frameHandle);
                frameHandle = null;
            }
            pendingSourceScene = null;
            syncScene(sceneName);
            activeSourceScene = null;
            return;
        }

        pendingSourceScene = sceneName;
        if (frameHandle === null) {
            frameHandle = requestAnimationFrame(flushPendingCamera);
        }
    }

    graph.on('plotly_relayouting', function (eventData) {
        synchronizeCamera(eventData, false);
    });
    graph.on('plotly_relayout', function (eventData) {
        synchronizeCamera(eventData, true);
    });
}());
"""


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
        help="Generate only this pair (default: generate all 21 pairs)",
    )
    parser.add_argument(
        "--context",
        choices=("constrained", "unconstrained", "midpoint"),
        default="constrained",
        help="Fixed values for all other weights (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        default="loss_weight_landscapes/3d",
        help="Directory for generated HTML plots (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        help="Output HTML path when generating a single pair with --pair",
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
    if args.output and args.pair is None:
        parser.error("--output requires --pair; use --output-dir for all pairs")
    return args


def make_pair_figure(args, pair, background, datasets):
    i, j = pair
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
    triangulation, train_losses, extended_losses = make_adaptive_surfaces(
        background,
        pair,
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

    titles = ("Training loss", "Validation loss (x scaled by 3)")
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "surface"}, {"type": "surface"}]],
        subplot_titles=titles,
        horizontal_spacing=0.06,
    )
    finite_surfaces = [surface[np.isfinite(surface)] for surface in display_grids]
    color_min = min(float(np.min(surface)) for surface in finite_surfaces)
    color_max = max(float(np.max(surface)) for surface in finite_surfaces)
    loss_min = min(float(np.min(surface)) for surface in finite_surfaces)
    loss_max = max(float(np.max(surface)) for surface in finite_surfaces)
    for index, (surface, title) in enumerate(zip(display_grids, titles), start=1):
        fig.add_trace(
            go.Surface(
                x=display_x,
                y=display_y,
                z=surface,
                name=title,
                colorscale="Viridis",
                cmin=color_min,
                cmax=color_max,
                showscale=index == 1,
                colorbar=dict(title="log10(MSE)") if index == 1 else None,
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
            ),
            row=1,
            col=index,
        )

    reference_weights = (CONSTRAINED_WEIGHTS, UNCONSTRAINED_WEIGHTS)
    reference_names = ("Constrained pair values", "Unconstrained pair values")
    for data_index, (dataset, title) in enumerate(zip(datasets, titles), start=1):
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
                showlegend=False,
                marker=dict(
                    size=7, color=("#ff365e", "#36d9ff"), symbol="diamond"
                ),
                text=reference_names,
                hovertemplate=(
                    "%{{text}}<br>{}: %{{x:.5g}}<br>{}: %{{y:.5g}}"
                    "<br>log10(MSE): %{{z:.5g}}<extra></extra>"
                ).format(WEIGHT_NAMES[i], WEIGHT_NAMES[j]),
            ),
            row=1,
            col=data_index,
        )

    fig.update_layout(
        title=dict(
            text=(
                "Loss landscape: {} vs {}<br>"
                "<sup>{} background; rotate either plot to rotate both, scroll to zoom</sup>"
            ).format(WEIGHT_NAMES[i], WEIGHT_NAMES[j], args.context),
            x=0.5,
        ),
        template="plotly_dark",
        paper_bgcolor="#050505",
        scene=dict(
            bgcolor="#050505",
            xaxis_title=WEIGHT_NAMES[i],
            yaxis_title=WEIGHT_NAMES[j],
            zaxis=dict(title="log10(MSE)", range=[loss_min, loss_max]),
            dragmode="orbit",
            camera=dict(eye=dict(x=1.65, y=1.65, z=1.2)),
            aspectmode="cube",
        ),
        scene2=dict(
            bgcolor="#050505",
            xaxis_title=WEIGHT_NAMES[i],
            yaxis_title=WEIGHT_NAMES[j],
            zaxis=dict(title="log10(MSE)", range=[loss_min, loss_max]),
            dragmode="orbit",
            camera=dict(eye=dict(x=1.65, y=1.65, z=1.2)),
            aspectmode="cube",
        ),
        margin=dict(l=0, r=0, t=105, b=0),
    )
    return fig, len(triangulation.x)


def main():
    args = parse_args()
    if args.context == "constrained":
        background = CONSTRAINED_WEIGHTS.copy()
    elif args.context == "unconstrained":
        background = UNCONSTRAINED_WEIGHTS.copy()
    else:
        background = (CONSTRAINED_WEIGHTS + UNCONSTRAINED_WEIGHTS) / 2.0

    data = make_data(args.sample_size)
    datasets = (
        (data["training"][0], data["training"][1]),
        (data["extended"][0], data["extended"][1]),
    )
    pairs = [args.pair] if args.pair is not None else list(
        itertools.combinations(range(len(WEIGHT_NAMES)), 2)
    )
    output_dir = Path(args.output_dir)
    for pair_index, pair in enumerate(pairs, start=1):
        i, j = pair
        print(
            "Generating pair {}/{}: {} vs {}".format(
                pair_index, len(pairs), WEIGHT_NAMES[i], WEIGHT_NAMES[j]
            ),
            flush=True,
        )
        fig, sample_count = make_pair_figure(args, pair, background, datasets)
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = output_dir / "loss_surface_3d_{}_vs_{}.html".format(
                WEIGHT_NAMES[i], WEIGHT_NAMES[j]
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(
            output_path,
            include_plotlyjs="directory",
            full_html=True,
            post_script=CAMERA_SYNC_SCRIPT,
        )
        print(
            "Wrote {} vs {} to {} ({} adaptive samples).".format(
                WEIGHT_NAMES[i],
                WEIGHT_NAMES[j],
                output_path.resolve(),
                sample_count,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
