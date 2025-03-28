# stacks

A `Stack`, abstractly, is a 2-tuple, where each element of the
tuple `(X, Y)` is an `n`-vector, satisfying the following conditions

    - `Y`: Sorted and strictly increasing. (Strict monotonicity implies `n`
        unique elements, too.)
    - `X`: Every element is strictly greater than zero.

A `Stack` can optionally have a "name" associated with it.


# stack arithmetic

Let `s1` and `s2` be two `Stack`s with names "rincewind" and "mortimer". The
**sum** of `s1` and `s2` is another `Stack`, `s3`, whose fundamental tuples
are:

    - `Y`: The union of `s1.Y` and `s2.Y`
    - `X`: The `i`-th element of `s3.X` is
        `sum(s1.X[s1.Y == s3.Y[i]]) + sum(s2.X[s2.Y == s3.Y[i]])`.

The name of `s3` is `rincewind+mortimer`.

If we ignore the names of `Stack`s for a second, the addition operation on
the set of `Stack`s is associative and commutative (although there is no
zero element because of our restriction on `Y`). When considering the names
of `Stack`s; the operation is associative but not commutative.


# fidelity

In terms of computation and arithmetic, a `Stack` is uniquely identified by
its `(X, Y)` vectors and its name. To the extent that `Stack`s are used for
data analysis/exploring, though, the provenance of each segment of the
`(X, Y)` vectors is meaningful. To leverage that information, each `Stack`
has a `sources` method that exposes the origin of each `Stack` segment.
For example,

```python
>>> from stacks import Stack

>>> s1 = Stack([3, 2, 5], [0.1, 0.2, 0.4], 'rincewind')
>>> s1.sources()
array(['rincewind', 'rincewind', 'rincewind'], dtype='<U9')

>>> s2 = Stack([1, 1, 1], [0.3, 0.5, 0.8], 'mortimer')
>>> s2.sources()
array(['mortimer', 'mortimer', 'mortimer'], dtype='<U9')

>>> crossover = s1 + s2
>>> crossover.sources()
array(['rincewind', 'rincewind', 'mortimer', 'rincewind', 'mortimer',
       'mortimer'], dtype='<U9')
```

The order of the names in `crossover.sources()` is determined by each component
stack's `Y` vector, and in this case amounts to splicing the two `Y`s together
and recording the location of `s1.Y` and `s2.Y` entries.

When two `Stack`s have overlapping `Y` vectors, their sum involves the collision
of the `X` entries. The `X` and `Y` vectors only keep track of the aggregates,
but under the hood the individual components are recorded. For example, suppose
we have
```python
>>> s1 = Stack([3, 2, 5], [0.1, 0.2, 0.4], 'rincewind')
>>> s2 = Stack([1, 1, 1], [0.3, 0.4, 0.8], 'mortimer')
```
The third element of `s1` and second element of `s2` will be aggregated together
in their sum:
```python
>>> s3 = s1 + s2
>>> s3.Y()
array([0.1, 0.2, 0.3, 0.4, 0.8])

>>> s3.X()
array([3, 2, 1, 6, 1])
```

If one wants to disaggregate the `X` and `Y` vectors that are grouped together,
pass `fine = True` to their respective methods:
```python
>>> s3.Y(fine=True)
array([0.1, 0.2, 0.3, 0.4, 0.4, 0.8])

>>> s3.X(fine=True)
array([3, 2, 1, 5, 1, 1])
```
Note, however, that the output of `s3.Y(fine=True)` is a valid `Y`-vector for a
`Stack` object if and only if the component `Stack`s have non-intersecting
`Y`-vectors themselves.

The `sources` method also accepts the `fine` parameter:
```python
>>> s3.sources()
array(['rincewind', 'rincewind', 'mortimer', 'rincewind+mortimer',
       'mortimer'], dtype='<U18')

>>> s3.sources(True)
array(['rincewind', 'rincewind', 'mortimer', 'rincewind', 'mortimer',
       'mortimer'], dtype='<U9')
```
As does the `cumulate` method:
```python
>>> s3.cumulate()
array([ 3,  5,  6, 12, 13])

>>> s3.cumulate(fine=True)
array([ 3,  5,  6, 11, 12, 13])
```

# visualization

I'm not sure how to properly integrate the notion of "backends" for plotting
`Stack` objects yet, and probably won't put the time toward learning it for
a while. In the meantime, there's a built-in `plot_data()` method that
transforms the underlying `X` and `Y` vectors, along with names, and returns
that information in dictionary to make plotting marginally easier.

![modo fig](https://raw.githubusercontent.com/hotzsauce/stacks/tree/main/figures/simple_stack_chart.png)

The following code snippet uses `pandas`/`matplotlib` to generate the figure
above. Handling the colors is crude here, but what are you gonna do.
```python
def plot_stack(st: Stack):
    # create a DataFrame with "x_tick", "width", "height", "name" columns
    df = pd.DataFrame.from_dict(st.plot_data())
    df["color"] = df.name.map(lambda n: "blue" if "rincewind" in n else "orange")

    # plot each group by name
    fig, ax = plt.subplots()
    for name, grp in df.groupby("name"):
        ax.bar(
            x=grp.x_tick,
            height=grp.height,
            width=grp.width,
            color=grp.color,
            edgecolor="white",
            label=name
        )

    ax.legend()
    plt.show()
```
The figure above uses the stack `s1 + s2`, where `s1 = Stack([3, 2, 5], [0.1,
0.2, 0.4], name='rincewind')` and `s2 = Stack([1, 1, 1], [0.3, 0.4, 0.5], name=
'mortimer')`.

As a proof of concept, in the `/figures` directory there is a sample of Hunt
Energy offers from August 2024, along with their `Stack`-produced figure, and
the script used to generate the figure.

![hunt fig](https://raw.githubusercontent.com/hotzsauce/stacks/tree/main/figures/hunt_aug2024_sample.png)

# notes

Regarding names, the default is the empty string, which is subsumed by other
names when adding `Stack`s. That is, if `s1` is named a nonempty string and
`s2` is unnamed, then the name of `s1 + s2` and `s2 + s1` is that of `s1`.
