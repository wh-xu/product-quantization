from pq import *
import warnings
import tqdm
from multiprocessing import Pool, cpu_count, sharedctypes


def _init(arr_to_populate, norms_square_to_populate):
    """Each pool process calls this initializer. Load the array to be populated into that process's global namespace"""
    global arr
    global norms_square
    arr = arr_to_populate
    norms_square = norms_square_to_populate


def arg_sort(arguments):
    """
    for q, sort the compressed items in 'compressed' by their distance,
    where distance is determined by 'metric'
    :param arguments: a tuple of (metric, compressed, q)
    :return:
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        compressed = np.ctypeslib.as_array(arr)
        norms_sqr = np.ctypeslib.as_array(norms_square)

    metric, q = arguments
    if metric == 'product':
        distances = [-np.dot(np.array(q).flatten(), np.array(center).flatten()) for center in compressed]
    elif metric == 'angular':
        distances = [
            - np.dot(np.array(q).flatten(), np.array(center).flatten()) / (np.linalg.norm(q) * np.linalg.norm(center))
            for center in compressed
        ]
    elif metric == 'sign':
        distances = [
            np.count_nonzero((q > 0) != (center > 0)) for center in compressed
        ]
    elif metric == 'euclid_norm':
        distances = [norm_sqr - 2.0 * np.dot(center, q) for center, norm_sqr in zip(compressed, norms_sqr)]
    else:
        distances = [np.linalg.norm(q - center) for center in compressed]

    distances = np.array(distances)

    top_k = min(131072, len(distances)-1)
    indices = np.argpartition(distances, top_k)[:top_k]
    return indices[np.argsort(distances[indices])]


def parallel_sort(metric, compressed, Q, X):
    """
    for each q in 'Q', sort the compressed items in 'compressed' by their distance,
    where distance is determined by 'metric'
    :param metric: euclid product
    :param compressed: compressed items, same dimension as origin data, shape(N * D)
    :param Q: queries, shape(len(Q) * D)
    :return:
    """
    tmp = np.ctypeslib.as_ctypes(compressed)
    shared_arr = sharedctypes.Array(tmp._type_, tmp, lock=False)
    norms_sqr = np.linalg.norm(X, axis=1) ** 2

    pool = Pool(processes=cpu_count(), initializer=_init, initargs=(shared_arr, norms_sqr))

    rank = list(tqdm.tqdm(pool.imap(arg_sort, zip([metric for _ in Q], Q), chunksize=4), total=len(Q)))
    pool.close()
    pool.join()
    return rank


class Sorter(object):
    def __init__(self, compressed, Q, X, metric='euclid'):
        self.Q = Q
        self.X = X

        self.topK = parallel_sort(metric, compressed, Q, X)

    def recall(self, G, T):
        t = min(T, len(self.topK[0]))
        assert len(self.topK) <= len(G)
        top_k = [len(np.intersect1d(G[i], self.topK[i][:T])) for i in range(len(self.topK))]
        return t, np.mean(top_k) / len(G[0])
