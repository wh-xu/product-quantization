# %%
import tqdm, math
import numpy as np
from transformer import *
import file_io

from pq import *
from opq import *
from aq import *
from pq_residual import *
from pq_norm import *
from pqx import *


# %%
def chunk_compress(pq, vecs):
    chunk_size = 1000000
    compressed_vecs = np.empty(shape=vecs.shape, dtype=np.float32)
    for i in tqdm.tqdm(range(math.ceil(len(vecs) / chunk_size))):
        compressed_vecs[i * chunk_size: (i + 1) * chunk_size, :] \
            = pq.compress(vecs[i * chunk_size: (i + 1) * chunk_size, :].astype(dtype=np.float32))
    return compressed_vecs

@nb.njit
def eval_error(x, gt):
    # Mean percentage error
    MPE = np.mean(np.abs((x - gt) / gt))
    MPE_max = np.max(np.abs((x - gt) / gt))
    # Mean absolute error
    MAE = np.mean(np.abs(x - gt))
    # Root mean squared error
    RMSE = np.sqrt(np.mean((x - gt) ** 2))
    return MPE, MAE, RMSE, MPE_max

def compute_distance(query, base, metric):
    @nb.njit
    def dist_product(x, y):
        return np.dot(-x, y.T)
    @nb.njit
    def dist_l2(x, y):
        dist = np.sum(x**2, axis=1)[:, np.newaxis] -2 * (x@y.T) + np.sum(y**2, axis=1)[:, np.newaxis].T
        return np.sqrt(dist)
    @nb.njit
    def dist_angular(x, y):
        return 1 - x@y.T / np.sqrt(np.sum(x**2, axis=1))[:, np.newaxis] / np.sqrt(np.sum(y**2, axis=1))[:, np.newaxis].T

    if metric == 'product':
        dist = dist_product(query, base)
    elif metric == 'euclidean':
        dist = dist_l2(query, base)
    elif metric == 'angular':
        dist = dist_angular(query, base)
    else:
        raise ValueError("unknown metric")
    return dist
    
def eval_dist(quantizer, base, query, metric, sample_size=10000):
    print('\t# compress base items')
    base_sample = base[0:sample_size, :]
    base_decode = quantizer.compress(base_sample)
    
    print('\t# compute distances')
    dist_acc = compute_distance(query, base_sample, metric)
    dist_approx = compute_distance(query, base_decode, metric)
    return dist_approx, dist_acc

def execute(pq, X, T, Q, G, metric, train_percentage=10):
    np.random.seed(123)
    print("# ranking metric {}".format(metric))
    print("# "+pq.class_message())
    
    train_size = int(X.shape[0] * train_percentage / 100)
    if T is None:
        pq.fit(X[:train_size].astype(dtype=np.float32), iter=20)
    else:
        pq.fit(T.astype(dtype=np.float32), iter=20)

    # print('# compress items')
    # compressed = chunk_compress(pq, X)
    
    print('# evaluate distance computation')
    dist_approx, dist_acc = eval_dist(pq, X, Q, metric) 
    MPE, MAE, RMSE, MPE_max = eval_error(dist_approx, dist_acc)
    return MPE, MAE, RMSE, MPE_max

# %%

dataset = 'netflix'
topk = 10
codebook = 8
Ks = 256

metric = 'product'
X, T, Q, G = file_io.loader(dataset, topk, metric, folder='data/')

# metric = 'euclidean'
# fname = './data/sift-128-euclidean.hdf5'

# metric = 'angular'
# fname = './data/glove-100-angular.hdf5'

# X, T, Q, G = file_io.hdf5_read(fname, topk)

# X /= np.linalg.norm(X)
# Q /= np.linalg.norm(Q)

print("# Parameters: dataset = {}, topK = {}, codebook = {}, Ks = {}, metric = {}".format(dataset, topk, codebook, Ks, metric))


# %%    
codebook = 32
Ks = 256

# 1. PQ
quantizer_pq = PQ(M=codebook, Ks=Ks)
MPE, MAE, RMSE, MPE_max = execute(quantizer_pq, X, T, Q, G, metric)
print("PQ: CB={}, K={}\n# MPE: {:.4f}\t MAE: {:.4f}\t RMSE: {:.4f}\t MPE_max: {:.4f}".format(codebook, Ks, MPE, MAE, RMSE, MPE_max))


# 2. OPQ
quantizer_opq = OPQ(M=codebook, Ks=Ks)
MPE, MAE, RMSE, MPE_max = execute(quantizer_opq, X, T, Q, G, metric)
print("OPQ: CB={}, K={}\n# MPE: {:.4f}\t MAE: {:.4f}\t RMSE: {:.4f}\t MPE_max: {:.4f}".format(codebook, Ks, MPE, MAE, RMSE, MPE_max))


# 4. NormPQ
quantizer_normPQ = NormPQ(n_percentile=Ks, quantize=PQ(M=codebook-1, Ks=Ks))
MPE, MAE, RMSE, MPE_max = execute(quantizer_normPQ, X, T, Q, G, metric)
print("normPQ: CB={}, K={}\n# MPE: {:.4f}\t MAE: {:.4f}\t RMSE: {:.4f}\t MPE_max: {:.4f}".format(codebook, Ks, MPE, MAE, RMSE, MPE_max))


# 5. NormOPQ
quantizer = OPQ(M=codebook-1, Ks=Ks)
quantizer_normOPQ = NormPQ(n_percentile=Ks, quantize=quantizer)
MPE, MAE, RMSE, MPE_max = execute(quantizer_normOPQ, X, T, Q, G, metric)
print("normOPQ: CB={}, K={}\n# MPE: {:.4f}\t MAE: {:.4f}\t RMSE: {:.4f}\t MPE_max: {:.4f}".format(codebook, Ks, MPE, MAE, RMSE, MPE_max))


RQ_configs = [[codebook, [Ks for i in range(codebook)]]]
# RQ_configs = [[codebook, [1024 if i<4 else (512 if i>=4 and i<8 else (256 if i>=8 and i<24 else (128 if i>=24 and i<28 else 64))) for i in range(codebook)]]]

for cb, Ks in RQ_configs:
    pqs = [PQ(M=1, Ks=k) for k in Ks]
    quantizer_rq = ResidualPQ(pqs=pqs)
    MPE, MAE, RMSE, MPE_max = execute(quantizer_rq, X, T, Q, G, metric)
    print("RQ: CB={}, K={}\n# MPE: {:.4f}\t MAE: {:.4f}\t RMSE: {:.4f}\t MPE_max: {:.4f}".format(codebook, Ks, MPE, MAE, RMSE, MPE_max))

    pqs = [PQ(M=1, Ks=Ks[i]) for i in range(cb-1)]
    quantizer = ResidualPQ(pqs=pqs)
    quantizer_normRQ = NormPQ(n_percentile=Ks[-1], quantize=quantizer)
    MPE, MAE, RMSE, MPE_max = execute(quantizer_normRQ, X, T, Q, G, metric)
    print("normRQ: CB={}, K={}\n# MPE: {:.4f}\t MAE: {:.4f}\t RMSE: {:.4f}\t MPE_max: {:.4f}".format(codebook, Ks, MPE, MAE, RMSE, MPE_max))

    