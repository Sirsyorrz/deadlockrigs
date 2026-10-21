"""Head mesh topology: welded vertices, UV islands, edges. python facegeo.py head.npz prints the islands."""
import sys
from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra


@dataclass
class Head:
    co: np.ndarray        # (n,3) unwelded positions
    tris: np.ndarray
    vmat: np.ndarray      # material per vertex
    weld: np.ndarray      # unwelded -> welded id
    wco: np.ndarray       # welded positions
    island: np.ndarray    # unwelded UV island id
    wgraph: object        # welded edge-length graph (csr)
    bones: dict
    wnrm: np.ndarray = None

    def islands_of(self, ids):
        return np.isin(self.island, ids)


def load(path: str) -> Head:
    d = np.load(path)
    co, tris = d['co'].astype(np.float64), d['tris']
    vmat = np.zeros(len(co), int)
    vmat[tris.ravel()] = np.repeat(d['mats'], 3)
    key = np.round(co / 1e-5).astype(np.int64)
    _, weld, = np.unique(key, axis=0, return_inverse=True)[:2]
    weld = weld.ravel()
    nw = weld.max() + 1
    wco = np.zeros((nw, 3))
    wco[weld] = co
    e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    n = len(co)
    g = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    _, island = connected_components(g, directed=False)
    we = weld[e]
    we = we[we[:, 0] != we[:, 1]]
    lens = np.linalg.norm(wco[we[:, 0]] - wco[we[:, 1]], axis=1)
    wg = coo_matrix((lens, (we[:, 0], we[:, 1])), shape=(nw, nw)).tocsr()
    wg = wg.maximum(wg.T)
    bones = {k[5:]: d[k] for k in d.files if k.startswith('bone_')}
    wnrm = np.zeros((nw, 3))
    np.add.at(wnrm, weld, d['nrm'])
    wnrm /= np.maximum(np.linalg.norm(wnrm, axis=1, keepdims=True), 1e-9)
    return Head(co, tris, vmat, weld, wco, island, wg, bones, wnrm)


def geodesic(h: Head, sources_welded, limit=np.inf) -> np.ndarray:
    return dijkstra(h.wgraph, directed=False, indices=np.unique(sources_welded), min_only=True, limit=limit)


if __name__ == '__main__':
    h = load(sys.argv[1])
    print('verts', len(h.co), 'welded', len(h.wco))
    for i in np.unique(h.island):
        m = h.island == i
        if m.sum() < 20:
            continue
        lo, hi = h.co[m].min(0), h.co[m].max(0)
        print('island', i, m.sum(), 'mat', np.unique(h.vmat[m]), np.round(lo, 3), np.round(hi, 3))
