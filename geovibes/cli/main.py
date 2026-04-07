"""GeoVibes CLI — interactive geospatial similarity search from the terminal."""

import click

from geovibes.cli.project import init_project, load_project, parse_bbox, parse_years


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """GeoVibes: satellite embedding search, labelling, and review."""
    pass


@cli.command()
@click.argument("name")
@click.option("--bbox", required=True, help="minx,miny,maxx,maxy in WGS84")
@click.option("--years", required=True, help="Comma-separated years (e.g. 2024,2025)")
@click.option("--res", default=80, type=int, help="Resolution in meters (10,20,40,80)")
@click.option("--target", default="", help="Description of search target")
def init(name, bbox, years, res, target):
    """Create a new GeoVibes project."""
    bbox_list = parse_bbox(bbox)
    years_list = parse_years(years)

    project_dir = init_project(
        name=name,
        bbox=bbox_list,
        years=years_list,
        resolution=res,
        target=target,
    )
    click.echo(f"Project created at {project_dir}")


@cli.command()
def load():
    """Load embeddings into the project database (aef-loader)."""
    config = load_project()
    click.echo(f"Loading embeddings for {config['name']}...")
    click.echo("Not yet implemented — requires aef-loader IFD support")


@cli.command()
@click.option("--positives", multiple=True, required=True, type=click.Path(exists=True))
@click.option("--negatives", multiple=True, type=click.Path(exists=True))
@click.option(
    "--classifier",
    default="linear-svm",
    type=click.Choice(["linear-svm", "xgboost"]),
)
@click.option("--threshold", default=0.5, type=float)
def train(positives, negatives, classifier, threshold):
    """Train a classifier on labelled embeddings."""
    config = load_project()
    click.echo(f"Training {classifier} for {config['name']}...")
    click.echo("Not yet implemented")


@cli.command()
@click.option("--threshold", default=0.5, type=float)
def infer(threshold):
    """Run inference over all embeddings."""
    config = load_project()
    click.echo(f"Running inference for {config['name']}...")
    click.echo("Not yet implemented")


@cli.command()
@click.option("--eps", default=500, type=int, help="DBSCAN eps in meters")
@click.option("--min-samples", default=2, type=int)
def cluster(eps, min_samples):
    """Cluster detections with DBSCAN."""
    config = load_project()
    click.echo(f"Clustering detections for {config['name']}...")
    click.echo("Not yet implemented")


@cli.command()
@click.option("--detections", required=True, type=click.Path(exists=True))
@click.option("--truth", required=True, type=click.Path(exists=True))
@click.option("--buffer", default=500, type=int, help="Match buffer in meters")
def validate(detections, truth, buffer):
    """Validate detections against ground truth."""
    click.echo("Not yet implemented")


@cli.command()
def tui():
    """Launch the interactive TUI."""
    from geovibes.cli.tui.app import GeoVibesTUI

    config = load_project()
    app = GeoVibesTUI(config)
    app.run()


if __name__ == "__main__":
    cli()
