"""cave-local robust multicut: no connectivity requirement on either side (gaps allowed)."""

import numpy as np
import pytest

from pychunkedgraph.graph.cutting import robust_mincut, run_multicut
from pychunkedgraph.graph.edges import Edges
from pychunkedgraph.graph.exceptions import PostconditionError
from pychunkedgraph.utils.general import in2d


def _edges(pairs, affs):
    pairs = np.array(pairs, dtype=np.uint64)
    both = np.concatenate([pairs, pairs[:, ::-1]])
    affs = np.concatenate([affs, affs]).astype(np.float32)
    return Edges(both[:, 0], both[:, 1], affinities=affs)


def _ids(*xs):
    return np.array(xs, dtype=np.uint64)


def _components(nodes, pairs, removed):
    removed = {tuple(e) for e in np.asarray(removed).tolist()}
    parent = {n: n for n in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        if (a, b) not in removed and (b, a) not in removed:
            parent[find(a)] = find(b)
    return {n: find(n) for n in nodes}


# Sinks 1 and 5 hang off source 3 on either side (1 - 2 - 3 - 4 - 5), with a weak branch 3 - 6 - 7
# to a second source. The sinks connect only through a source: upstream path augmentation refuses.
PAIRS = [(1, 2), (2, 3), (3, 4), (4, 5), (3, 6), (6, 7)]
AFFS = [0.9, 0.2, 0.3, 0.8, 0.9, 0.9]


def test_upstream_refuses_disconnected_sinks():
    with pytest.raises(Exception, match="irreparably|overlap|separate|Mincut"):
        run_multicut(_edges(PAIRS, AFFS), _ids(3, 7), _ids(1, 5), robust=False)


def test_robust_separates_with_gaps():
    cut = run_multicut(_edges(PAIRS, AFFS), _ids(3, 7), _ids(1, 5), robust=True)
    comp = _components(range(1, 8), PAIRS, cut)
    assert comp[3] == comp[7]
    assert comp[1] not in (comp[3],) and comp[5] not in (comp[3],)
    assert comp[1] != comp[5]  # the sinks end up in two pieces: allowed
    cut_set = {tuple(sorted(e)) for e in np.asarray(cut).tolist()}
    assert cut_set == {(2, 3), (3, 4)}  # the weakest edges on each side


def test_robust_never_cuts_infinite_edges():
    pairs = [(1, 2), (2, 3)]
    with pytest.raises(PostconditionError):
        robust_mincut(np.array(pairs, np.uint64), np.array([np.inf, np.inf]), [1], [3])
    cut = robust_mincut(np.array(pairs, np.uint64), np.array([np.inf, 0.5]), [1], [3])
    assert {tuple(sorted(e)) for e in cut.tolist()} == {(2, 3)}


def test_robust_off_by_env(monkeypatch):
    monkeypatch.setenv("PCG_ROBUST_MULTICUT", "0")
    with pytest.raises(Exception, match="irreparably|overlap|separate|Mincut"):
        run_multicut(_edges(PAIRS, AFFS), _ids(3, 7), _ids(1, 5))


def test_in2d_accepts_transposed_arrays():
    a = np.array([[1, 2], [3, 4]], dtype=np.uint64)
    b = np.array([[1, 3], [2, 4]], dtype=np.uint64).T  # rows (1, 2), (3, 4); not C-contiguous
    assert not b.flags["C_CONTIGUOUS"]
    assert in2d(a, b).tolist() == [True, True]


# cave-local: a terminal with no local edge used to shift the raw-graph vertex ids of every later
# terminal (wrong terminals, or "Invalid vertex index"). It is now a precondition failure; the robust
# fallback of a real edit refuses such points too, with a message instead of a crash.
@pytest.mark.parametrize("sources", [[10, 99], [5, 10]])
def test_isolated_terminal_is_a_precondition_not_a_shifted_vertex(sources):
    from pychunkedgraph.graph.cutting import LocalMincutGraph
    from pychunkedgraph.graph.exceptions import PreconditionError

    edges = _edges([[10, 20], [20, 30], [30, 40], [40, 50]], [0.9, 0.9, 0.1, 0.9])
    with pytest.raises(PreconditionError, match="no edges in the local graph"):
        LocalMincutGraph(
            edges.get_pairs(), edges.affinities, _ids(*sources), _ids(50), False, True,
            disallow_isolating_cut=True,
        )
    with pytest.raises(PreconditionError, match="not on edges of the local graph"):
        run_multicut(edges, _ids(*sources), _ids(50), robust=True)
    with pytest.raises(PreconditionError):
        run_multicut(edges, _ids(*sources), _ids(50), split_preview=True)
