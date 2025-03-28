from __future__ import annotations

import numpy as np



class Stack(object):
    """
    A `Stack`, abstractly, is a 2-tuple, where each element of the
    tuple `(X, Y)` is an `n`-vector, satisfying the following conditions:
        - `X`: Every element is strictly greater than zero.
        - `Y`: Sorted and strictly increasing. (Strict monotonicity implies
            `n` unique elements, too.)
    A `Stack` can optionally have a "name" associated with it.

    Let `s1` and `s2` be two `Stack`s with names "rincewind" and "mortimer".
    The **sum** of `s1` and `s2` is another `Stack`, `s3`, whose fundamental
    tuples are:
        - `Y`: The union of `s1.Y` and `s2.Y`
        - `X`: The `i`-th element of `s3.X` is
            `sum(s1.X[s1.Y == s3.Y[i]]) + sum(s2.X[s2.Y == s3.Y[i]])`.
    The name of `s3` is `rincewind+mortimer`.

    If we ignore the names of `Stack`s for a second, the addition operation on
    the set of `Stack`s is associative and commutative (although there is no
    zero element because of our restriction on `Y`). When considering the names
    of `Stack`s; the operation is associative but not commutative.

    # Notes

    On the topic of names, the default is the empty string, which is subsumed
    by other names when adding `Stacks`. That is, if `s1` is named a nonempty
    string and `s2` is unnamed, then the name of `s1 + s2` and `s2 + s1` is
    that of `s1`.
    """

    # =====
    # Instantiation
    # =====

    def __init__(
        self,
        X: Iterable[Number],
        Y: Iterable[Number],
        name: str = "",
        sources: np.ndarray = None,
        strict: bool = True,
        ufunc: np.ufunc = np.add, # already over-engineering stuff
    ):
        X = np.asarray(X)
        Y = np.asarray(Y)
        _validate_stack_vectors(X, Y, strict)

        self.f = ufunc
        self.name = name

        self._X = X
        self._Y = Y

        if sources is None:
            self._sources = np.asarray([{name: x} for x in self._X])
        else:
            self._sources = sources

    @classmethod
    def from_blocks(
        cls,
        X: np.ndarray,
        Y: np.ndarray,
        **init_kwargs
    ):
        """
        Create a `Stack` from blocks of `X` and `Y` values

        # Example
        Y = [[ 0.95 14.99 29.99]
             [ 0.     nan   nan]
             [  nan  1.99  1.99]
             [ 0.     nan   nan]
             [  nan  1.99  1.99]]

        X = [[1.5 2.5 2.5]
             [9.9 nan nan]
             [nan 2.4 4. ]
             [9.9 nan nan]
             [nan 2.4 4. ]]

        stack = Stack.from_blocks(X, Y)
        [ 0.    0.95  1.99 14.99 29.99]
        [3.96 4.26 6.82 7.32 7.82]
        """
        p = Y.shape
        q = X.shape

        if p != q:
            raise ValueError(
                f"dims of `Y` block ({p}) do not match `X` block ({q})"
            )

        if len(p) == 2:
            Y = Y.flatten()
            X = X.flatten()
        elif len(p) == 1:
            pass
        else:
            raise ValueError("`X` and `Y` must be 1- or 2-dim arrays")

        # drop any nans and flatten each array
        mask = ~np.isnan(Y)
        Y_fl = Y[mask]
        X_fl = X[mask]

        # sort both x and y by y
        idx = np.argsort(Y_fl)
        Y_fl = Y_fl[idx]
        X_fl = X_fl[idx]

        # find unique heights and their initial indices; group accordingly
        Y_uniq, indices = np.unique(Y_fl, return_index=True)
        X_sums = np.add.reduceat(X_fl, indices)

        return Stack(X_sums, Y_uniq, **init_kwargs)

    # =====
    # Dunder Methods
    # =====
    def __add__(self, other: Stack):
        if type(self) != type(other):
            raise TypeError(f"cannot add `{type(other)}` objects to `Stack`s")

        Xl, Yl = self.vectors(fine=False)
        Xr, Yr = other.vectors(fine=False)
        X, Y, il, ir = _add_stacks_with_zipper(Xl, Yl, Xr, Yr)

        name = _join_stack_names(self, other)
        sources = _combine_sources(self._sources, other._sources, il, ir)

        return Stack(X, Y, name, sources)

    def __len__(self):
        return len(self._X)

    def __radd__(self, other: Any):
        if type(self) != type(other):
            raise TypeError(f"cannot add `Stack`s to `{type(other)} objects`")
        return other.__add__(self)

    def __repr__(self):
        n = len(self)
        x0, xn = self._X[0], self._X[n-1]

        if self.name:
            return f"Stack({self.name}, {x0}...{xn}; n={n})"
        else:
            return f"Stack({x0}...{xn}; n={n})"

    # =====
    # Fundamental Methods
    # =====
    def cumulate(self, fine: bool = False):
        """
        Calculate and return the cumulated `X` vector
        """
        if fine:
            it = (v for s_dict in self._sources for v in s_dict.values())
            X = np.fromiter(it, dtype=float)
        else:
            X = self._X

        return self.f.accumulate(X)

    def sources(self, fine: bool = False):
        if fine:
            it = (s for s_dict in self._sources for s in s_dict.keys())
            l = max(max(map(len, self.name.split(ADD_DELIM))), 1)
            return np.fromiter(it, dtype=f"<U{l}")
        else:
            it = (_join_names(s_dict.keys()) for s_dict in self._sources)
            l = max(len(self.name), 1)
            return np.fromiter(it, dtype=f"<U{l}")

    def vectors(self, fine: bool = False):
        return (self.X(fine), self.Y(fine))

    def X(self, fine: bool = False):
        if fine:
            it = (v for s_dict in self._sources for v in s_dict.values())
            return np.fromiter(it, dtype=self._X.dtype)
        else:
            return self._X

    def Y(self, fine: bool = False):
        """
        Having a `y`-vector that is the same size as `X(fine=True)` will
        probably be useful; however it does mean the corresponding
        `Y(fine=True)` vector will not be a proper vertical vector of a `Stack`
        as it would not be strictly increasing
        """
        if fine:
            it = (y for s, y in zip(self._sources, self._Y) for _ in s.values())
            return np.fromiter(it, dtype=self._Y.dtype)
        else:
            return self._Y

    # =====
    # Visualization
    # =====
    def plot_data(self, fine: bool = True):
        """
        Return a dictionary with the following items:
            - 'x_tick': An `n`-vector of the midpoints of the bars to plot
            - 'width': An `n`-vector of the widths of each bar
            - 'height': An `n`-vector of the heights of each bar
            - 'name': An `n`-vector of the names for each bar
        """
        Y = self.Y(fine)
        X = self.cumulate(fine)

        # compute the centered x-ticks and widths of the bars
        Z = np.concatenate(([0], X))
        X = (Z[1:] + Z[:-1]) / 2
        W = np.diff(Z)

        if fine:
            names = self.sources(True)
        else:
            names = np.full(len(Y), self.name)

        return {"x_tick": X, "width": W, "height": Y, "name": names}




# =====
# Utilities
# =====

ADD_DELIM = "+"



def _add_stacks(
    Xl: np.ndarray,
    Yl: np.ndarray,
    Xr: np.ndarray,
    Yr: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interlace two Stacks `(Xl, Yl)` and `(Xr, Yr)` into the expected `(X, Y)`
    summed Stack. The vectors `X` and `Y` are returned
    """
    X, Y, _, _ = _add_stacks_with_zipper(Xl, Yl, Xr, Yr)
    return X, Y

def _add_stacks_with_zipper(
    Xl: np.ndarray,
    Yl: np.ndarray,
    Xr: np.ndarray,
    Yr: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Interlace two Stacks `(Xl, Yl)` and `(Xr, Yr)` into the expected `(X, Y)`
    summed Stack. The indexes used to map `Yl` and `Yr` into `Y` are also
    returned
    """

    # this is an O(n+m) operation equivalent to `np.union1d(...)`, which is
    # O((n+m)log(n+m)). Inputs must be sorted, though (which is our situation)
    Y = np.unique(np.concatenate((Yl, Yr)))

    # initialize the sum's X-values
    X = np.zeros(len(Y), np.result_type(Xl, Xr))

    # maps from Yl -> Y and Yr -> Y space
    il = np.searchsorted(Y, Yl)
    ir = np.searchsorted(Y, Yr)

    # fill em all in
    X[il] += Xl
    X[ir] += Xr

    return X, Y, il, ir

def _combine_sources(
    Sl: np.ndarray[dict[str, float]],
    Sr: np.ndarray[dict[str, float]],
    il: np.ndarray,
    ir: np.ndarray,
):
    n = max(np.max(il), np.max(ir)) + 1
    S = np.array([{} for _ in range(n)])

    for i, l in enumerate(il):
        S[l] = _combine_source_dict(S[l], Sl[i])

    for i, r in enumerate(ir):
        S[r] = _combine_source_dict(S[r], Sr[i])

    return S

def _combine_source_dict(
    sl: dict[str, float],
    sr: dict[str, float]
) -> dict[str, float]:
    for key_r, value_r in sr.items():
        if key_r in sl:
            sl[key_r] += value_r
        else:
            sl[key_r] = value_r
    return sl

def _join_stack_names(sl: Stack, sr: Stack) -> str:
    return _join_names((sl.name, sr.name))
    if nl.name and nr.name:
        if nl.name == nr.name:
            return nl.name
        return ADD_DELIM.join((nl.name, nr.name))
    elif nl.name:
        return nl.name
    elif nr.name:
        return nr.name
    else:
        return ""

def _join_names(names: Iterable[str]) -> str:
    seen = set()
    out = []

    for name in names:
        if name and (name not in seen):
            seen.add(name)
            out.append(name)

    if not out:
        return ""

    if len(out) == 1:
        return out[0]

    return ADD_DELIM.join(out)

def _validate_stack_vectors(X: np.ndarray, Y: np.ndarray, strict: bool):
    """
    Make the following checks on the vectors:
        - `X`: Every element is strictly greater than zero
        - `Y`: Monotonically increasing (this handles sorted & unique too)
        - Both: They are both actually vectors (as opposed to nd-arrays), and
            of the same size.
    """
    if (len(X.shape) > 1) or (len(Y.shape) > 1):
        raise ValueError(f"`Stack` arrays must both be vectors")

    if X.shape != Y.shape:
        x = X.shape[0]
        y = Y.shape[0]

        msg = (
            "`Stack` arrays are not of the same size"
            f"X ({X.shape[0]}) != Y ({Y.shape[0]})"
        )
        raise ValueError(msg)

    if X.shape[0] < 1:
        raise ValueError("`Stack` vectors must be nonempty")

    if np.any(X <= 0):
        raise ValueError(f"the `X` vector of `Stack` has non-positive elements")

    if np.any(np.diff(Y) <= 0):
        msg = (
            "the `Y` vector of `Stack` is either not sorted or not "
            "monotonically increasing"
        )
        raise ValueError(msg)
