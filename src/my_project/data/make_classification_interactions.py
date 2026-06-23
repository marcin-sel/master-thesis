from __future__ import annotations

import numpy as np
from sklearn.datasets._samples_generator import _generate_hypercube
from sklearn.utils import Bunch, check_random_state
from sklearn.utils import shuffle as util_shuffle


def make_classification(
    n_samples=100,
    n_features=20,
    *,
    n_informative=2,
    n_redundant=2,
    n_repeated=0,
    n_classes=2,
    n_clusters_per_class=2,
    weights=None,
    flip_y=0.01,
    class_sep=1.0,
    hypercube=True,
    shift=0.0,
    scale=1.0,
    shuffle=True,
    random_state=None,
    return_X_y=True,
):
    """Generate a random n-class classification problem.

    This initially creates clusters of points normally distributed (std=1)
    about vertices of an ``n_informative``-dimensional hypercube with sides of
    length ``2*class_sep`` and assigns an equal number of clusters to each
    class. It introduces interdependence between these features and adds
    various types of further noise to the data.

    Without shuffling, ``X`` horizontally stacks features in the following
    order: the primary ``n_informative`` features, followed by ``n_redundant``
    linear combinations of the informative features, followed by ``n_repeated``
    duplicates, drawn randomly with replacement from the informative and
    redundant features. The remaining features are filled with random noise.
    Thus, without shuffling, all useful features are contained in the columns
    ``X[:, :n_informative + n_redundant + n_repeated]``.

    Read more in the :ref:`User Guide <sample_generators>`.

    Parameters
    ----------
    n_samples : int, default=100
        The number of samples.

    n_features : int, default=20
        The total number of features. These comprise ``n_informative``
        informative features, ``n_redundant`` redundant features,
        ``n_repeated`` duplicated features and
        ``n_features-n_informative-n_redundant-n_repeated`` useless features
        drawn at random.

    n_informative : int, default=2
        The number of informative features. Each class is composed of a number
        of gaussian clusters each located around the vertices of a hypercube
        in a subspace of dimension ``n_informative``. For each cluster,
        informative features are drawn independently from  N(0, 1) and then
        randomly linearly combined within each cluster in order to add
        covariance. The clusters are then placed on the vertices of the
        hypercube.

    n_redundant : int, default=2
        The number of redundant features. These features are generated as
        random linear combinations of the informative features.

    n_repeated : int, default=0
        The number of duplicated features, drawn randomly from the informative
        and the redundant features.

    n_classes : int, default=2
        The number of classes (or labels) of the classification problem.

    n_clusters_per_class : int, default=2
        The number of clusters per class.

    weights : array-like of shape (n_classes,) or (n_classes - 1,),\
              default=None
        The proportions of samples assigned to each class. If None, then
        classes are balanced. Note that if ``len(weights) == n_classes - 1``,
        then the last class weight is automatically inferred.
        More than ``n_samples`` samples may be returned if the sum of
        ``weights`` exceeds 1. Note that the actual class proportions will
        not exactly match ``weights`` when ``flip_y`` isn't 0.

    flip_y : float, default=0.01
        The fraction of samples whose class is assigned randomly. Larger
        values introduce noise in the labels and make the classification
        task harder. Note that the default setting flip_y > 0 might lead
        to less than ``n_classes`` in y in some cases.

    class_sep : float, default=1.0
        The factor multiplying the hypercube size.  Larger values spread
        out the clusters/classes and make the classification task easier.

    hypercube : bool, default=True
        If True, the clusters are put on the vertices of a hypercube. If
        False, the clusters are put on the vertices of a random polytope.

    shift : float, ndarray of shape (n_features,) or None, default=0.0
        Shift features by the specified value. If None, then features
        are shifted by a random value drawn in [-class_sep, class_sep].

    scale : float, ndarray of shape (n_features,) or None, default=1.0
        Multiply features by the specified value. If None, then features
        are scaled by a random value drawn in [1, 100]. Note that scaling
        happens after shifting.

    shuffle : bool, default=True
        Shuffle the samples and the features.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    return_X_y : bool, default=True
        If True, a tuple ``(X, y)`` instead of a Bunch object is returned.

        .. versionadded:: 1.7

    Returns
    -------
    data : :class:`~sklearn.utils.Bunch` if `return_X_y` is `False`.
        Dictionary-like object, with the following attributes.

        DESCR : str
            A description of the function that generated the dataset.
        parameter : dict
            A dictionary that stores the values of the arguments passed to the
            generator function.
        feature_info : list of len(n_features)
            A description for each generated feature.
        X : ndarray of shape (n_samples, n_features)
            The generated samples.
        y : ndarray of shape (n_samples,)
            An integer label for class membership of each sample.

        .. versionadded:: 1.7

    (X, y) : tuple if ``return_X_y`` is True
        A tuple of generated samples and labels.

    See Also
    --------
    make_blobs : Simplified variant.
    make_multilabel_classification : Unrelated generator for multilabel tasks.

    Notes
    -----
    The algorithm is adapted from Guyon [1] and was designed to generate
    the "Madelon" dataset.

    References
    ----------
    .. [1] I. Guyon, "Design of experiments for the NIPS 2003 variable
           selection benchmark", 2003.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(random_state=42)
    >>> X.shape
    (100, 20)
    >>> y.shape
    (100,)
    >>> list(y[:5])
    [np.int64(0), np.int64(0), np.int64(1), np.int64(1), np.int64(0)]
    """
    generator = check_random_state(random_state)

    # Count features, clusters and samples
    if n_informative + n_redundant + n_repeated > n_features:
        raise ValueError(
            "Number of informative, redundant and repeated "
            "features must sum to less than the number of total"
            " features"
        )
    # Use log2 to avoid overflow errors
    if n_informative < np.log2(n_classes * n_clusters_per_class):
        msg = "n_classes({}) * n_clusters_per_class({}) must be"
        msg += " smaller or equal 2**n_informative({})={}"
        raise ValueError(
            msg.format(n_classes, n_clusters_per_class, n_informative, 2**n_informative)
        )

    if weights is not None:
        # we define new variable, weight_, instead of modifying user defined parameter.
        if len(weights) not in [n_classes, n_classes - 1]:
            raise ValueError(
                "Weights specified but incompatible with number of classes."
            )
        if len(weights) == n_classes - 1:
            if isinstance(weights, list):
                weights_ = weights + [1.0 - sum(weights)]
            else:
                weights_ = np.resize(weights, n_classes)
                weights_[-1] = 1.0 - sum(weights_[:-1])
        else:
            weights_ = weights.copy()
    else:
        weights_ = [1.0 / n_classes] * n_classes

    n_random = n_features - n_informative - n_redundant - n_repeated
    n_clusters = n_classes * n_clusters_per_class

    # Distribute samples among clusters by weight
    n_samples_per_cluster = [
        int(n_samples * weights_[k % n_classes] / n_clusters_per_class)
        for k in range(n_clusters)
    ]

    for i in range(n_samples - sum(n_samples_per_cluster)):
        n_samples_per_cluster[i % n_clusters] += 1

    # Initialize X and y
    X = np.zeros((n_samples, n_features))
    y = np.zeros(n_samples, dtype=int)

    # Build the polytope whose vertices become cluster centroids
    centroids = _generate_hypercube(n_clusters, n_informative, generator).astype(
        float, copy=False
    )
    centroids *= 2 * class_sep
    centroids -= class_sep
    if not hypercube:
        centroids *= generator.uniform(size=(n_clusters, 1))
        centroids *= generator.uniform(size=(1, n_informative))

    # Initially draw informative features from the standard normal
    X[:, :n_informative] = generator.standard_normal(size=(n_samples, n_informative))

    # Create each cluster; a variant of make_blobs
    stop = 0
    for k, centroid in enumerate(centroids):
        start, stop = stop, stop + n_samples_per_cluster[k]
        y[start:stop] = k % n_classes  # assign labels
        X_k = X[start:stop, :n_informative]  # slice a view of the cluster

        A = 2 * generator.uniform(size=(n_informative, n_informative)) - 1
        X_k[...] = np.dot(X_k, A)  # introduce random covariance

        X_k += centroid  # shift the cluster to a vertex

    # Create redundant features
    if n_redundant > 0:
        B = 2 * generator.uniform(size=(n_informative, n_redundant)) - 1
        X[:, n_informative : n_informative + n_redundant] = np.dot(
            X[:, :n_informative], B
        )

    # Repeat some features
    n = n_informative + n_redundant
    if n_repeated > 0:
        indices = ((n - 1) * generator.uniform(size=n_repeated) + 0.5).astype(np.intp)
        X[:, n : n + n_repeated] = X[:, indices]

    # Fill useless features
    if n_random > 0:
        X[:, -n_random:] = generator.standard_normal(size=(n_samples, n_random))

    # Randomly replace labels
    if flip_y >= 0.0:
        flip_mask = generator.uniform(size=n_samples) < flip_y
        y[flip_mask] = generator.randint(n_classes, size=flip_mask.sum())

    # Randomly shift and scale
    if shift is None:
        shift = (2 * generator.uniform(size=n_features) - 1) * class_sep
    X += shift

    if scale is None:
        scale = 1 + 100 * generator.uniform(size=n_features)
    X *= scale

    indices = np.arange(n_features)
    if shuffle:
        # Randomly permute samples
        X, y = util_shuffle(X, y, random_state=generator)

        # Randomly permute features
        generator.shuffle(indices)
        X[:, :] = X[:, indices]

    if return_X_y:
        return X, y

    # feat_desc describes features in X
    feat_desc = ["random"] * n_features
    for i, index in enumerate(indices):
        if index < n_informative:
            feat_desc[i] = "informative"
        elif n_informative <= index < n_informative + n_redundant:
            feat_desc[i] = "redundant"
        elif n <= index < n + n_repeated:
            feat_desc[i] = "repeated"

    parameters = {
        "n_samples": n_samples,
        "n_features": n_features,
        "n_informative": n_informative,
        "n_redundant": n_redundant,
        "n_repeated": n_repeated,
        "n_classes": n_classes,
        "n_clusters_per_class": n_clusters_per_class,
        "weights": weights,
        "flip_y": flip_y,
        "class_sep": class_sep,
        "hypercube": hypercube,
        "shift": shift,
        "scale": scale,
        "shuffle": shuffle,
        "random_state": random_state,
        "return_X_y": return_X_y,
    }

    bunch = Bunch(
        DESCR=make_classification.__doc__,
        parameters=parameters,
        feature_info=feat_desc,
        X=X,
        y=y,
    )

    return bunch
