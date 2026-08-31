from pathlib import Path
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def test_optional_dependency_contract():
    data = tomllib.loads(
        Path("pyproject.toml").read_text(encoding="utf-8")
    )

    base = data["project"]["dependencies"]
    extras = data["project"]["optional-dependencies"]

    assert not any(
        dep.startswith("sentence-transformers") for dep in base
    )
    assert not any(
        dep.startswith("google-antigravity") for dep in base
    )

    assert any(
        dep.startswith("sentence-transformers")
        for dep in extras["memory"]
    )

    assert any(
        dep.startswith("google-antigravity")
        for dep in extras["antigravity"]
    )

    assert set(extras["full"]) == set(
        extras["memory"] + extras["antigravity"]
    )
