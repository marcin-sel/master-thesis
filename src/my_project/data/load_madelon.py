import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

MADELON_URL = "https://archive.ics.uci.edu/static/public/171/madelon.zip"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "my_project"
CACHE_FILENAME = "madelon.zip"


def _get_archive_bytes(
    url: str,
    timeout: float,
    cache_dir: Path,
    force_download: bool,
) -> bytes:
    """Return the Madelon archive bytes, using a local cache when available."""
    cache_path = cache_dir / CACHE_FILENAME

    if cache_path.exists() and not force_download:
        return cache_path.read_bytes()

    response = requests.get(url, timeout=timeout)
    response.raise_for_status()

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(response.content)
    return response.content


def _find_member(archive: zipfile.ZipFile, suffix: str) -> str:
    """Return the first archive member whose name ends with ``suffix``."""
    suffix = suffix.lower()
    try:
        return next(
            name for name in archive.namelist() if name.lower().endswith(suffix)
        )
    except StopIteration as exc:
        raise FileNotFoundError(
            f"No member ending with {suffix!r} found in the Madelon archive."
        ) from exc


def _read_matrix(archive: zipfile.ZipFile, suffix: str) -> pd.DataFrame:
    """Read a whitespace-delimited, header-less matrix from the archive."""
    with archive.open(_find_member(archive, suffix)) as file:
        return pd.read_csv(file, sep=r"\s+", header=None)


def load_madelon_data(
    *,
    url: str = MADELON_URL,
    timeout: float = 120.0,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    force_download: bool = False,
) -> dict[str, pd.DataFrame | pd.Series | list[str]]:
    """Load the UCI Madelon dataset as a single merged train + validation set.

    Madelon is the artificial classification benchmark from the NIPS 2003
    feature-selection challenge: 500 continuous features (only 20 informative,
    the rest redundant or pure noise) and a binary target. The official archive
    ships the data pre-split into ``train``, ``valid`` and ``test`` parts, but
    the ``test`` labels are not public. This loader downloads the archive,
    **ignores the train/valid split and concatenates both labelled parts** into
    one dataset (the unlabelled ``test`` part is dropped).

    The archive is cached on disk (see ``cache_dir``): it is downloaded only on
    the first call, and reused from the cache on subsequent calls.

    The return value mirrors the shape of
    :func:`~my_project.data.generate_synthetic_data_pairwise_interaction.generate_pairwise_interaction_data`:
    a dictionary with a named feature frame ``X`` and a named target
    ``y`` (plus a little metadata), so both can be used interchangeably by
    downstream tooling.

    Parameters
    ----------
    url:
        Location of the Madelon ``.zip`` archive. Defaults to the UCI mirror.
    timeout:
        Timeout (seconds) for the HTTP download.
    cache_dir:
        Directory used to cache the downloaded archive. Defaults to
        ``~/.cache/my_project``. The archive is fetched from ``url`` only if it
        is not already present there.
    force_download:
        If ``True``, re-download the archive even when a cached copy exists
        (and refresh the cache).

    Returns
    -------
    dict[str, pandas.DataFrame | pandas.Series | list[str]]
        Dictionary with keys ``"X"``, ``"y"`` and ``"feature_names"``.

    X : pandas.DataFrame
        Feature matrix of shape ``(2600, 500)`` (2000 train + 600 valid rows),
        with columns named ``x1..x500``.
    y : pandas.Series
        Binary target named ``"y"``, with the original ``{-1, 1}`` labels
        remapped to ``{0, 1}``.
    feature_names : list[str]
        The ``x1..x500`` column names, for convenience.
    """
    archive_bytes = _get_archive_bytes(
        url=url,
        timeout=timeout,
        cache_dir=Path(cache_dir),
        force_download=force_download,
    )

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        X_train = _read_matrix(archive, "madelon_train.data")
        y_train = _read_matrix(archive, "madelon_train.labels").squeeze("columns")

        X_valid = _read_matrix(archive, "madelon_valid.data")
        y_valid = _read_matrix(archive, "madelon_valid.labels").squeeze("columns")

    X = pd.concat([X_train, X_valid], ignore_index=True)
    y = pd.concat([y_train, y_valid], ignore_index=True)

    feature_names = [f"x{i + 1}" for i in range(X.shape[1])]
    X.columns = feature_names

    y = (y == 1).astype(int)
    y = pd.Series(y.to_numpy(), name="y")

    return {
        "X": X,
        "y": y,
        "feature_names": feature_names,
    }


def generate_madelon_data(
    n_samples: int | None = None,
    *,
    random_state: int | None = None,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    force_download: bool = False,
    **_ignored,
) -> dict[str, pd.DataFrame | pd.Series | list]:
    """Load Madelon behind the synthetic-generator interface.

    Drop-in replacement for the synthetic generators (e.g.
    :func:`~my_project.data.generate_synthetic_data_pairwise_interaction.generate_pairwise_interaction_data`):
    it returns the same dict contract (``"X"``, ``"y"``, ``"true_interactions"``)
    so the same experiment pipeline can train on the real Madelon benchmark.

    Madelon is a *fixed* dataset (2600 labelled rows), so the generator-style
    arguments are handled as follows:

    - ``n_samples``: if given, a **stratified** subsample of exactly that many
      rows is drawn (using ``random_state``); ``None`` returns all 2600 rows.
      Requesting more than 2600 rows raises ``ValueError`` (the split arithmetic
      in the training pipeline assumes the frame has exactly ``n_samples`` rows).
    - ``random_state``: seeds the subsample only.
    - any other keyword (``cov``, ``n_informative``, ``interactions``, ...) is
      accepted and ignored, so the generic ``generate(**GENERATOR_KWARGS)``
      wrapper can stay generator-agnostic.

    ``true_interactions`` is an empty list: Madelon has 20 informative features
    combined non-linearly, but no known *pairwise* ground-truth interaction
    graph, so there is no oracle edge set to return.

    Returns
    -------
    dict
        ``{"X", "y", "coef", "cov", "true_interactions"}``. ``coef`` and ``cov``
        are ``None`` (no generative parameters exist for a real dataset); the
        keys are kept for parity with the synthetic generators.
    """
    data = load_madelon_data(cache_dir=cache_dir, force_download=force_download)
    X = data["X"]
    y = data["y"]

    if n_samples is not None:
        if n_samples > len(X):
            raise ValueError(
                f"Madelon has only {len(X)} labelled rows; cannot draw "
                f"n_samples={n_samples}."
            )
        if n_samples < len(X):
            from sklearn.model_selection import train_test_split

            X, _, y, _ = train_test_split(
                X,
                y,
                train_size=n_samples,
                stratify=y,
                random_state=random_state,
            )
            X = X.reset_index(drop=True)
            y = y.reset_index(drop=True)

    return {
        "X": X,
        "y": y,
        "coef": None,
        "cov": None,
        "true_interactions": [],
    }
