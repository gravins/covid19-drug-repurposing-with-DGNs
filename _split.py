import numpy as np

from sklearn.model_selection._split import (
    _BaseKFold, BaseShuffleSplit, _validate_shuffle_split, _approximate_mode
)

from sklearn.utils import check_random_state
from sklearn.utils.validation import column_or_1d, check_array
from sklearn.utils.multiclass import type_of_target

import warnings


DA_NSPLIT_WARNING = (
    "Randomness is used only as group tie-breaker. Using more than "
    "one split may produce equal or very similar partitions."
)


class DAKFold(_BaseKFold):
    """Distribution-Aware K-Folds cross-validator

    Provides train/test indices to split data in train/test sets. Split
    dataset into k consecutive folds, such that the samples in each fold
    maintain the same distribution of the values in ``group``.

    Each fold is then used once as a validation while the k - 1 remaining
    folds form the training set.

    Read more in the :ref:`User Guide <cross_validation>`.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.

    shuffle : boolean, optional
        Whether to shuffle the data before splitting into batches. Notice that
        randomization is used only for tie-breaking values in ``groups``, and
        has no effect it has all distinct values.

    random_state : int, RandomState instance or None, optional, default=None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`. Only used when ``shuffle`` is True. This should be left
        to None if ``shuffle`` is False.

    Examples
    --------
    >>> import numpy as np
    >>> from dacv import DAKFold
    >>> X = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    >>> groups = X**2  # [ 1,  4,  9, 16, 25, 36, 49, 64]
    >>> dakf = DAKFold(n_splits=2)
    >>> dakf.get_n_splits(X, groups=groups)
    2
    >>> print(dakf)
    DAKFold(n_splits=2, random_state=None, shuffle=False)
    >>> for train_index, test_index in dakf.split(X, groups=groups):
    ...     print("TRAIN:", train_index, "TEST:", test_index)
    ...     X_train, X_test = X[train_index], X[test_index]
    ...     groups_train, groups_test = groups[train_index], groups[test_index]
    TRAIN: [1 3 5 7] TEST: [0 2 4 6]
    TRAIN: [0 2 4 6] TEST: [1 3 5 7]

    Notes
    -----
    The first ``n_samples % n_splits`` folds have size
    ``n_samples // n_splits + 1``, other folds have size
    ``n_samples // n_splits``, where ``n_samples`` is the number of samples.

    Randomized CV splitters may return different results for each call of
    split. You can make the results identical by setting ``random_state``
    to an integer.

    See also
    --------
    DAStratifiedKFold
        Takes group information into account to avoid building folds with
        imbalanced class distributions (for binary or multiclass
        classification tasks).
    """

    def __init__(self, n_splits=5, shuffle=False, random_state=None):            
        super().__init__(n_splits, shuffle, random_state)
    
    def _make_test_folds(self, X, y=None, groups=None):
        gs = np.asarray(groups)
        n_samples = gs.shape[0]
        gs = gs.reshape((n_samples, -1))

        if self.shuffle:
            rng = check_random_state(self.random_state)
            idx = rng.permutation(n_samples)
            gs = np.concatenate((gs, idx.reshape((-1, 1))), axis=-1)

        test_folds = np.empty(n_samples, dtype=np.int)
        idx = np.lexsort(gs.T[::-1])
        test_folds[idx] = np.arange(n_samples, dtype=np.int) % self.n_splits

        return test_folds

    def _iter_test_masks(self, X=None, y=None, groups=None):
        test_folds = self._make_test_folds(X, y, groups)
        for i in range(self.n_splits):
            yield test_folds == i

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like, shape (n_samples,)
            The target variable for supervised learning problems.

        groups : array-like, shape (n_samples, n_attributes)
            Attribute associated to each sample, whose value distribution is
            preserved in each fold.

            If ``n_attributes > 1``, then the values of each attribute
            following the first one is used as tie-break for the preceding
            attribute values.

            Note that providing ``groups`` is sufficient to generate the
            splits. and hence ``np.zeros(n_samples)`` may be used as a
            placeholder for ``X`` instead of actual training data.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.

        Notes
        -----
        Randomized CV splitters may return different results for each call of
        split. You can make the results identical by setting ``random_state``
        to an integer.
        """

        return super().split(X, y, groups)


class DAStratifiedKFold(DAKFold):
    """Distribution-Aware Stratified K-Folds cross-validator

    Provides train/test indices to split data in train/test sets.

    This cross-validation object is a variation of DAKFold that returns
    stratified folds. The folds are made by preserving the percentage of
    samples for each class.

    Read more in the :ref:`User Guide <cross_validation>`.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.

    shuffle : boolean, optional
        Whether to shuffle the data before splitting into batches. Notice that
        randomization is used only for tie-breaking values in ``groups``, and
        has no effect it has all distinct values.

    random_state : int, RandomState instance or None, optional, default=None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`. Only used when ``shuffle`` is True. This should be left
        to None if ``shuffle`` is False.

    Examples
    --------
    >>> import numpy as np
    >>> from dacv import DAStratifiedKFold
    >>> X = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    >>> y = np.array([0, 0, 0, 1, 1, 0, 1, 1])
    >>> groups = X**2  # [ 1,  4,  9, 16, 25, 36, 49, 64]
    >>> daskf = DAStratifiedKFold(n_splits=2)
    >>> daskf.get_n_splits(X, y, groups)
    2
    >>> print(daskf)
    DAStratifiedKFold(n_splits=2, random_state=None, shuffle=False)
    >>> for train_index, test_index in daskf.split(X, y, groups):
    ...     print("TRAIN:", train_index, "TEST:", test_index)
    ...     X_train, X_test = X[train_index], X[test_index]
    ...     y_train, y_test = y[train_index], y[test_index]
    TRAIN: [1 4 5 7] TEST: [0 2 3 6]
    TRAIN: [0 2 3 6] TEST: [1 4 5 7]

    Notes
    -----
    The implementation is designed to:

    * Generate test sets such that all contain the same distribution of
      classes, or as close as possible.
    * Be invariant to class label: relabelling ``y = ["Happy", "Sad"]`` to
      ``y = [1, 0]`` should not change the indices generated.
    * Preserve order dependencies in the dataset ordering, when
      ``shuffle=False``: all samples from class k in some test set were
      contiguous in y, or separated in y by samples from classes other than k.
    * Generate test sets where the smallest and largest differ by at most one
      sample.

    """

    def _make_test_folds(self, X, y=None, groups=None):
        y = np.asarray(y)
        type_of_target_y = type_of_target(y)
        allowed_target_types = ('binary', 'multiclass')
        if type_of_target_y not in allowed_target_types:
            raise ValueError(
                'Supported target types are: {}. Got {!r} instead.'.format(
                    allowed_target_types, type_of_target_y))

        y = column_or_1d(y)
        n_samples = y.shape[0]
        unique_y = np.unique(y)

        gs = np.asarray(groups)
        gs = gs.reshape((n_samples, -1))

        test_folds = np.empty(n_samples, dtype=np.int)
        shift = 0
        
        for label in unique_y:
            mask = (y == label)
            test_folds[mask] = (super()._make_test_folds(None, None, gs[mask])
                                + shift) % self.n_splits
            shift += mask.sum()
        
        return test_folds

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like, shape (n_samples,)
            The target variable for supervised learning problems.
            Stratification is done based on the y labels.

        groups : array-like, shape (n_samples, n_attributes)
            Attribute associated to each sample, whose value distribution is
            preserved in each fold.

            If ``n_attributes > 1``, then the values of each attribute
            following the first one is used as tie-break for the preceding
            attribute values.

            Note that providing ``groups`` is sufficient to generate the
            splits. and hence ``np.zeros(n_samples)`` may be used as a
            placeholder for ``X`` instead of actual training data.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.

        Notes
        -----
        Randomized CV splitters may return different results for each call of
        split. You can make the results identical by setting ``random_state``
        to an integer.
        """

        y = check_array(y, ensure_2d=False, dtype=None)
        return super().split(X, y, groups)


class DASplit(BaseShuffleSplit):
    """Distribution-Aware cross-validator

    Yields indices to split data into training and test sets, such that the
    samples in therein have the same distribution with respect to the values
    in ``group``.

    Note: contrary to other randomized cross-validation splits, this method
    is likely to return the same folds for each repetitions whenever the
    values in ``groups`` are distinct. Thus, it is suggested not to perform
    more than 1 repetitions (``n_splits = 1``).

    Read more in the :ref:`User Guide <cross_validation>`.

    Parameters
    ----------
    n_splits : int (default 1)
        Number of re-shuffling & splitting iterations.

    test_size : float, int, None, default=None
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test samples. If None, the value is set to the
        complement of the train size. If ``train_size`` is also None, it will
        be set to 0.1.

    train_size : float, int, or None, default=None
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split. If
        int, represents the absolute number of train samples. If None,
        the value is automatically set to the complement of the test size.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Examples
    --------
    >>> import numpy as np
    >>> from dacv import DASplit
    >>> X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> groups = X**2  # [ 1,  4,  9, 16, 25, 36, 49, 64, 81]
    >>> das = DASplit(n_splits=5, test_size=3, random_state=0)
    >>> das.get_n_splits(X, groups=groups)
    5
    >>> print(das)
    DASplit(n_splits=5, random_state=0, test_size=3, train_size=None)
    >>> for train_index, test_index in das.split(X, groups=groups):
    ...     print("TRAIN:", train_index, "TEST:", test_index)
    TRAIN: [0 2 3 5 6 8] TEST: [1 4 7]
    TRAIN: [0 2 3 5 6 8] TEST: [1 4 7]
    TRAIN: [0 2 3 5 6 8] TEST: [1 4 7]
    TRAIN: [0 2 3 5 6 8] TEST: [1 4 7]
    TRAIN: [0 2 3 5 6 8] TEST: [1 4 7]
    >>> groups = X // 2  # [0, 1, 1, 2, 2, 3, 3, 4, 4]
    >>> for train_index, test_index in das.split(X, groups=groups):
    ...     print("TRAIN:", train_index, "TEST:", test_index)
    TRAIN: [0 1 3 6 5 8] TEST: [2 4 7]
    TRAIN: [0 2 3 5 6 8] TEST: [1 4 7]
    TRAIN: [0 2 4 5 6 7] TEST: [1 3 8]
    TRAIN: [0 2 4 5 6 8] TEST: [1 3 7]
    TRAIN: [0 1 4 6 5 7] TEST: [2 3 8]
    """

    def __init__(self, n_splits=1, test_size="default", train_size=None,
                 random_state=None):
        if n_splits > 1:
            warnings.warn(DA_NSPLIT_WARNING, Warning)

        super().__init__(n_splits, test_size, train_size, random_state)
    
    def _iter_indices(self, X, y=None, groups=None):
        gs = np.asarray(groups)
        n_samples = gs.shape[0]
        gs = gs.reshape((n_samples, -1))
        rng = check_random_state(self.random_state)
        perm = rng.permutation(n_samples).reshape((-1, 1))
        gs = np.concatenate([gs, perm], axis=-1)
        _, n_test = _validate_shuffle_split(n_samples,
                                            self.test_size,
                                            self.train_size)

        for _ in range(self.n_splits):
            perm = np.lexsort(gs.T[::-1])
            idx = np.linspace(0, n_samples, n_test,
                              endpoint=False, dtype=int)
            idx += (n_samples - idx[-1] - 1) // 2
            test_mask = np.zeros(n_samples, dtype=np.bool)
            test_mask[idx] = True

            yield perm[~test_mask], perm[test_mask]

            gs[:, -1] = rng.permutation(n_samples)

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like, shape (n_samples,)
            The target variable for supervised learning problems.

        groups : array-like, shape (n_samples, n_attributes)
            Attribute associated to each sample, whose value distribution is
            preserved in each fold.

            If ``n_attributes > 1``, then the values of each attribute
            following the first one is used as tie-break for the preceding
            attribute values.

            Note that providing ``groups`` is sufficient to generate the
            splits. and hence ``np.zeros(n_samples)`` may be used as a
            placeholder for ``X`` instead of actual training data.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.

        Notes
        -----
        Randomized CV splitters may return different results for each call of
        split. You can make the results identical by setting ``random_state``
        to an integer.
        """

        return super().split(X, y, groups)


class DAStratifiedSplit(DASplit):
    """Distribution-Aware Stratified cross-validator

    Provides train/test indices to split data in train/test sets.

    This cross-validation object is a merge of DAStratifiedKFold and DASplit,
    which returns stratified randomized folds. The folds are made by
    preserving the percentage of samples for each class and also the
    distribution of the values in ``group``.

    Note: contrary to other randomized cross-validation splits, this method
    is likely to return the same folds for each repetitions whenever the
    values in ``groups`` are distinct. Thus, it is suggested not to perform
    more than 1 repetitions (``n_splits = 1``).

    Read more in the :ref:`User Guide <cross_validation>`.

    Parameters
    ----------
    n_splits : int, default 1
        Number of re-shuffling & splitting iterations.

    test_size : float, int, None, optional (default=None)
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test samples. If None, the value is set to the
        complement of the train size. If ``train_size`` is also None, it will
        be set to 0.1.

    train_size : float, int, or None, default is None
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split. If
        int, represents the absolute number of train samples. If None,
        the value is automatically set to the complement of the test size.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Examples
    --------
    >>> import numpy as np
    >>> from dacv import DAStratifiedSplit
    >>> X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1])
    >>> groups = X**2  # [ 1,  4,  9, 16, 25, 36, 49, 64, 81]
    >>> dass = DAStratifiedSplit(n_splits=5, test_size=3, random_state=0)
    >>> dass.get_n_splits(X, y, groups=groups)
    5
    >>> print(dass)
    DAStratifiedSplit(n_splits=5, random_state=0, test_size=3, train_size=None)
    >>> for train_index, test_index in dass.split(X, groups=groups):
    ...     print("TRAIN:", train_index, "TEST:", test_index)
    TRAIN: [0 2 3 5 6 8] TEST: [1 4 7]
    TRAIN: [0 2 3 5 6 8] TEST: [1 4 7]
    TRAIN: [0 2 3 5 6 8] TEST: [1 4 7]
    TRAIN: [0 2 3 5 6 8] TEST: [1 4 7]
    TRAIN: [0 2 3 5 6 8] TEST: [1 4 7]
    >>> groups = X // 2  # [0, 1, 1, 2, 2, 3, 3, 4, 4]
    >>> for train_index, test_index in das.split(X, groups=groups):
    ...     print("TRAIN:", train_index, "TEST:", test_index)
    TRAIN: [0 1 4 5 6 8] TEST: [2 3 7]
    TRAIN: [0 2 4 5 6 8] TEST: [1 3 7]
    TRAIN: [0 2 4 5 6 7] TEST: [1 3 8]
    TRAIN: [0 1 3 5 6 7] TEST: [2 4 8]
    TRAIN: [0 2 4 5 6 8] TEST: [1 3 7]
    """

    def _iter_indices(self, X, y=None, groups=None):
        y = np.asarray(y)
        type_of_target_y = type_of_target(y)
        allowed_target_types = ('binary', 'multiclass')
        if type_of_target_y not in allowed_target_types:
            raise ValueError(
                'Supported target types are: {}. Got {!r} instead.'.format(
                    allowed_target_types, type_of_target_y))

        y = column_or_1d(y)
        n_samples = y.shape[0]
        unique_y, y_indices = np.unique(y, return_inverse=True)
        class_counts = np.bincount(y_indices)

        gs = np.asarray(groups)
        gs = gs.reshape((n_samples, -1))
        _, n_test = _validate_shuffle_split(n_samples,
                                            self.test_size,
                                            self.train_size)

        rng = check_random_state(self.random_state)

        for _ in range(self.n_splits):
            idx = np.arange(n_samples, dtype=np.int)
            t_i = _approximate_mode(class_counts, n_test, rng)
            train_indices, test_indices = [], []

            for label, test_size in zip(unique_y, t_i):
                lbl_mask = (y == label)
                lbl_idx = idx[lbl_mask]
                das = DASplit(self.n_splits, test_size, random_state=rng)
                lbl_train, lbl_test = next(das._iter_indices(None, None, 
                                                             gs[lbl_mask]))
                train_indices.append(lbl_idx[lbl_train])
                test_indices.append(lbl_idx[lbl_test])

            yield np.concatenate(train_indices), np.concatenate(test_indices)

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like, shape (n_samples,)
            The target variable for supervised learning problems.
            Stratification is done based on the y labels.

        groups : array-like, shape (n_samples, n_attributes)
            Attribute associated to each sample, whose value distribution is
            preserved in each fold.

            If ``n_attributes > 1``, then the values of each attribute
            following the first one is used as tie-break for the preceding
            attribute values.

            Note that providing ``groups`` is sufficient to generate the
            splits. and hence ``np.zeros(n_samples)`` may be used as a
            placeholder for ``X`` instead of actual training data.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.

        Notes
        -----
        Randomized CV splitters may return different results for each call of
        split. You can make the results identical by setting ``random_state``
        to an integer.
        """

        y = check_array(y, ensure_2d=False, dtype=None)
        return super().split(X, y, groups)
