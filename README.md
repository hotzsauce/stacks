# stacks

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
