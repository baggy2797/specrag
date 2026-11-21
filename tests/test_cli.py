from typer.testing import CliRunner

from specrag.cli import app


def test_version() -> None:
    r = CliRunner().invoke(app, ["--version"])
    assert r.exit_code == 0 and "0.1.0" in r.output
