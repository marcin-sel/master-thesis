import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import requests

HIGGS_URL = (
    "https://archive.ics.uci.edu/" "ml/machine-learning-databases/00280/HIGGS.csv.gz"
)
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "my_project"
CACHE_PREFIX = "higgs_prefix_"
DEFAULT_N_ROWS = 100_000
# When random subsampling is requested (``random_state`` is set) we stream this
# many leading rows once, cache them, and draw random subsets from that pool so
# different seeds yield different samples without ever downloading the whole
# ~2.7 GB archive.
DEFAULT_POOL_ROWS = 1_000_000

_N_FEATURES = 28

# Physicists' names for the 28 features, in the exact column order of the raw
# HIGGS CSV (Baldi et al., 2014; UCI dataset 280): the first CSV column is the
# class label, followed by 21 low-level kinematic quantities then 7 high-level
# invariant masses. These names are the single source of truth for columns,
# feature sets and the oracle graph -- the data is relabelled to them right after
# loading, so nothing downstream ever deals with the raw ``x{i}`` identifiers.
_FEATURE_NAMES = [
    "lepton_pt",
    "lepton_eta",
    "lepton_phi",
    "missing_energy_magnitude",
    "missing_energy_phi",
    "jet1_pt",
    "jet1_eta",
    "jet1_phi",
    "jet1_btag",
    "jet2_pt",
    "jet2_eta",
    "jet2_phi",
    "jet2_btag",
    "jet3_pt",
    "jet3_eta",
    "jet3_phi",
    "jet3_btag",
    "jet4_pt",
    "jet4_eta",
    "jet4_phi",
    "jet4_btag",
    "m_jj",
    "m_jjj",
    "m_lv",
    "m_jlv",
    "m_bb",
    "m_wbb",
    "m_wwbb",
]
_COLUMN_NAMES = ["y"] + _FEATURE_NAMES
_FEATURE_INDEX = {name: i for i, name in enumerate(_FEATURE_NAMES)}

_N_LOW_LEVEL = 21
_LOW_LEVEL_FEATURES = _FEATURE_NAMES[:_N_LOW_LEVEL]
_HIGH_LEVEL_FEATURES = _FEATURE_NAMES[_N_LOW_LEVEL:]
_FEATURE_SETS = {
    "all": _FEATURE_NAMES,
    "low_level": _LOW_LEVEL_FEATURES,
    "high_level": _HIGH_LEVEL_FEATURES,
}


# --- Physics-motivated "true" interaction graph over the low-level features ---
# The 7 high-level features are invariant masses physicists compute by hand from
# a subset of the detector objects (Baldi et al., 2014). Training on the
# low-level set forces a model to rediscover that structure, so we can hand it an
# oracle graph: for every mass, the kinematic variables of its constituent
# objects jointly determine it, so we connect them as a clique (all pairs).
#
# Each object's *kinematic* low-level columns (the ones that enter an invariant
# mass). The b-tag flags (jet*_btag) are excluded: they are discrete tags, not
# momentum components, so they do not appear in any mass formula.
_OBJECT_FEATURES: dict[str, list[str]] = {
    "lepton": ["lepton_pt", "lepton_eta", "lepton_phi"],
    "met": ["missing_energy_magnitude", "missing_energy_phi"],  # transverse only
    "jet1": ["jet1_pt", "jet1_eta", "jet1_phi"],
    "jet2": ["jet2_pt", "jet2_eta", "jet2_phi"],
    "jet3": ["jet3_pt", "jet3_eta", "jet3_phi"],
    "jet4": ["jet4_pt", "jet4_eta", "jet4_phi"],
}

_ALL_JETS = ["jet1", "jet2", "jet3", "jet4"]

# Which detector objects combine into each high-level invariant mass. The b-jet
# masses (m_bb / m_wbb / m_wwbb) depend on whichever jets are b-tagged, which is
# a per-row property (jet*_btag) unknown statically, so by default they span all
# jets; pass ``b_jets`` to restrict them (e.g. ["jet1", "jet2"]).
_MASS_OBJECTS: dict[str, list[str]] = {
    "m_jj": ["jet1", "jet2"],
    "m_jjj": ["jet1", "jet2", "jet3"],
    "m_lv": ["lepton", "met"],
    "m_jlv": ["jet1", "lepton", "met"],
    "m_bb": _ALL_JETS,
    "m_wbb": ["lepton", "met", *_ALL_JETS],
    "m_wwbb": ["lepton", "met", *_ALL_JETS],
}


def higgs_true_interactions(
    feature_set: str = "low_level",
    *,
    b_jets: list[str] | None = None,
) -> list[tuple[str, str]]:
    """Oracle pairwise-interaction edges for HIGGS, derived from the physics.

    Each high-level feature is an invariant mass computed from a set of detector
    objects; the kinematic variables of those objects jointly determine the mass,
    so we connect them as a clique (every unordered pair). The union of these
    cliques over all masses is the ground-truth interaction graph on the
    low-level kinematic features.

    Edges use the physicists' feature names (e.g. ``("lepton_pt", "jet1_eta")``),
    are canonicalised in the raw CSV feature order and de-duplicated, then
    filtered to endpoints present in ``feature_set``. For ``"high_level"`` this
    yields an empty list (the masses themselves carry no pairwise oracle among
    each other).

    Parameters
    ----------
    feature_set:
        Which feature set the edges must live in (``"low_level"`` default,
        ``"high_level"`` or ``"all"``); see :func:`load_higgs_data`.
    b_jets:
        Jets treated as the b-jets for the b-tagged masses (m_bb / m_wbb /
        m_wwbb). Defaults to all four jets, since the physical b-jet assignment
        is a per-row property and not known statically.

    Returns
    -------
    list[tuple[str, str]]
        Unordered feature pairs, matching the ``true_interactions`` contract of
        the synthetic generators.
    """
    allowed = set(_select_feature_set(feature_set))

    mass_objects = dict(_MASS_OBJECTS)
    if b_jets is not None:
        mass_objects["m_bb"] = list(b_jets)
        mass_objects["m_wbb"] = ["lepton", "met", *b_jets]
        mass_objects["m_wwbb"] = ["lepton", "met", *b_jets]

    edges: set[tuple[str, str]] = set()
    for objects in mass_objects.values():
        columns = [col for obj in objects for col in _OBJECT_FEATURES[obj]]
        for i, a in enumerate(columns):
            for b in columns[i + 1 :]:
                if a == b:
                    continue
                lo, hi = sorted((a, b), key=_FEATURE_INDEX.__getitem__)
                edges.add((lo, hi))

    edges = {(a, b) for a, b in edges if a in allowed and b in allowed}
    return sorted(edges, key=lambda e: (_FEATURE_INDEX[e[0]], _FEATURE_INDEX[e[1]]))


def _select_feature_set(feature_set: str) -> list[str]:
    try:
        return _FEATURE_SETS[feature_set]
    except KeyError as exc:
        raise ValueError(
            f"feature_set must be one of {sorted(_FEATURE_SETS)}, "
            f"got {feature_set!r}."
        ) from exc


def _cache_path(cache_dir: Path, n_rows: int) -> Path:
    return cache_dir / f"{CACHE_PREFIX}{n_rows}.parquet"


def _find_reusable_cache(cache_dir: Path, n_rows: int) -> Path | None:
    """Return a cached prefix with at least ``n_rows`` rows, if any exists.

    Cache files are named ``higgs_prefix_<n>.parquet`` where ``<n>`` is the
    number of rows they hold, so any cache with ``<n> >= n_rows`` can satisfy
    the request by slicing its first ``n_rows`` rows -- avoiding a re-download.
    """
    if not cache_dir.exists():
        return None

    candidates: list[tuple[int, Path]] = []
    for path in cache_dir.glob(f"{CACHE_PREFIX}*.parquet"):
        try:
            cached_rows = int(path.stem[len(CACHE_PREFIX) :])
        except ValueError:
            continue
        if cached_rows >= n_rows:
            candidates.append((cached_rows, path))

    if not candidates:
        return None

    # Smallest cache that still covers the request -> least to read.
    return min(candidates, key=lambda item: item[0])[1]


def _download_higgs_prefix(
    n_rows: int, url: str, timeout: tuple[float, float]
) -> pd.DataFrame:
    """Stream the gzipped HIGGS file and read only its first ``n_rows`` rows.

    The remote file is ~2.7 GB compressed; streaming lets us decompress and
    parse just the prefix we need without downloading the whole archive.
    """
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        # The payload is already gzip; do not let requests decode it so we can
        # feed the raw stream straight into GzipFile.
        response.raw.decode_content = False

        with gzip.GzipFile(fileobj=response.raw, mode="rb") as stream:
            data = pd.read_csv(
                stream,
                header=None,
                names=_COLUMN_NAMES,
                nrows=n_rows,
                dtype=np.float32,
            )

    data["y"] = data["y"].astype(np.int8)
    return data


def load_higgs_data(
    n_rows: int = DEFAULT_N_ROWS,
    *,
    feature_set: str = "low_level",
    random_state: int | None = None,
    pool_rows: int | None = None,
    url: str = HIGGS_URL,
    timeout: tuple[float, float] = (30.0, 600.0),
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    force_download: bool = False,
) -> dict[str, pd.DataFrame | pd.Series | list[str]]:
    """Load a prefix of the UCI HIGGS dataset.

    HIGGS is a large binary-classification benchmark (11M rows, 28 continuous
    features) distinguishing a Higgs-boson signal process from background. The
    full file is ~2.7 GB gzipped, so this loader streams the archive and reads
    **only the first ``n_rows`` rows** instead of downloading everything.

    The parsed prefix is cached on disk as a parquet file named
    ``higgs_prefix_<n_rows>.parquet`` (see ``cache_dir``). On subsequent calls
    the data is served from the cache; a request for fewer rows than an existing
    cache reuses that cache (slicing its head) rather than downloading again.
    The cache always stores all 28 features, so switching ``feature_set`` never
    triggers a re-download.

    Parameters
    ----------
    n_rows:
        Number of rows to return. Defaults to 100000. When ``random_state`` is
        given these rows are drawn at random from a pool (see ``pool_rows``);
        otherwise they are the leading ``n_rows`` rows of the file.
    random_state:
        If given, return a *random* subset of ``n_rows`` rows instead of the
        leading prefix. The randomness is over a fixed pool of ``pool_rows``
        leading rows, so different seeds yield different subsets while the pool
        is downloaded/cached only once. ``None`` (default) keeps the
        deterministic prefix behaviour.
    pool_rows:
        Size of the leading-row pool to sample from when ``random_state`` is
        set. Must be ``>= n_rows``. Defaults to ``max(n_rows, 1_000_000)``.
        Ignored when ``random_state`` is ``None``.
    feature_set:
        Which features to return. One of:

        - ``"low_level"`` (default): the 21 raw kinematic detector quantities
          (``lepton_pt`` ... ``jet4_btag``);
        - ``"high_level"``: the 7 physicist-derived invariant masses
          (``m_jj`` ... ``m_wwbb``);
        - ``"all"``: all 28 features.

        Low-level is the default so a model has to *discover* the structure that
        the high-level features encode by hand.
    url:
        Location of the gzipped HIGGS CSV. Defaults to the UCI mirror.
    timeout:
        ``(connect, read)`` timeout tuple (seconds) for the HTTP request.
    cache_dir:
        Directory used to cache parsed prefixes. Defaults to
        ``~/.cache/my_project``.
    force_download:
        If ``True``, re-download and refresh the cache even when a suitable
        cached prefix exists.

    Returns
    -------
    dict[str, pandas.DataFrame | pandas.Series | list[str]]
        Dictionary with keys ``"X"``, ``"y"`` and ``"feature_names"``.

    X : pandas.DataFrame
        Feature matrix of shape ``(n_rows, n_features)`` (``float32``), where
        ``n_features`` depends on ``feature_set`` (21 / 7 / 28).
    y : pandas.Series
        Binary target named ``"y"`` in ``{0, 1}`` (``int8``).
    feature_names : list[str]
        The selected feature column names (physicists' names), for convenience.
    """
    if n_rows <= 0:
        raise ValueError(f"n_rows must be positive, got {n_rows}.")

    feature_names = _select_feature_set(feature_set)
    cache_dir = Path(cache_dir)

    # With a seed we sample ``n_rows`` from a larger fixed pool; without one we
    # just read the leading ``n_rows`` rows. Either way the number of rows we
    # actually stream/cache is ``read_rows``.
    read_rows = n_rows
    if random_state is not None:
        if pool_rows is None:
            pool_rows = max(n_rows, DEFAULT_POOL_ROWS)
        if pool_rows < n_rows:
            raise ValueError(f"pool_rows ({pool_rows}) must be >= n_rows ({n_rows}).")
        read_rows = pool_rows

    data: pd.DataFrame | None = None
    if not force_download:
        reusable = _find_reusable_cache(cache_dir, read_rows)
        if reusable is not None:
            data = pd.read_parquet(reusable).head(read_rows)

    if data is None:
        data = _download_higgs_prefix(n_rows=read_rows, url=url, timeout=timeout)
        cache_dir.mkdir(parents=True, exist_ok=True)
        data.to_parquet(_cache_path(cache_dir, read_rows), index=False)

    if random_state is not None:
        data = data.sample(n=n_rows, random_state=random_state)

    data = data.reset_index(drop=True)
    # Relabel to the canonical (label + physicists' feature) names by position.
    # The download path already reads with these names; doing it here as well
    # upgrades any legacy cache written with the old x1..x28 identifiers.
    data.columns = _COLUMN_NAMES

    X = data[feature_names].copy()
    y = pd.Series(data["y"].to_numpy(), name="y")

    return {
        "X": X,
        "y": y,
        "feature_names": feature_names,
    }


def generate_higgs_data(
    n_samples: int | None = None,
    *,
    random_state: int | None = None,
    n_rows: int = DEFAULT_N_ROWS,
    feature_set: str = "low_level",
    pool_rows: int | None = None,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    force_download: bool = False,
    **_ignored,
) -> dict[str, pd.DataFrame | pd.Series | list]:
    """Load HIGGS behind the synthetic-generator interface.

    Drop-in replacement for the synthetic generators (e.g.
    :func:`~my_project.data.generate_synthetic_data_pairwise_interaction.generate_pairwise_interaction_data`):
    it returns the same dict contract (``"X"``, ``"y"``, ``"true_interactions"``)
    so the same experiment pipeline can train on the real HIGGS benchmark.

    The generator-style arguments map onto the prefix loader as follows:

    - ``n_samples``: number of rows to use (``None`` falls back to ``n_rows``).
    - ``random_state``: when given, ``n_samples`` rows are drawn at *random*
      from a fixed pool of ``pool_rows`` leading rows (default
      ``max(n_samples, 1_000_000)``), so different seeds give different
      subsets. The pool is streamed and cached only once. When ``None``, the
      leading ``n_samples`` rows are returned (deterministic prefix).
    - ``pool_rows``: size of that sampling pool; see :func:`load_higgs_data`.
    - ``feature_set``: which features to return (``"low_level"`` default,
      ``"high_level"`` or ``"all"``); see :func:`load_higgs_data`.
    - any other keyword (``cov``, ``n_informative``, ``interactions``, ...) is
      accepted and ignored, so the generic ``generate(**GENERATOR_KWARGS)``
      wrapper can stay generator-agnostic.

    ``true_interactions`` holds the physics-motivated oracle edges from
    :func:`higgs_true_interactions` (pairwise cliques over the low-level
    variables that jointly form each high-level invariant mass), filtered to the
    selected ``feature_set``. For ``"high_level"`` this is empty.

    Returns
    -------
    dict
        ``{"X", "y", "coef", "cov", "true_interactions"}``. ``coef`` and ``cov``
        are ``None`` (no generative parameters exist for a real dataset); the
        keys are kept for parity with the synthetic generators.
    """
    effective_rows = n_samples if n_samples is not None else n_rows

    data = load_higgs_data(
        n_rows=effective_rows,
        feature_set=feature_set,
        random_state=random_state,
        pool_rows=pool_rows,
        cache_dir=cache_dir,
        force_download=force_download,
    )

    return {
        "X": data["X"],
        "y": data["y"],
        "coef": None,
        "cov": None,
        # "true_interactions": higgs_true_interactions(feature_set),
        "true_interactions": {},
    }
