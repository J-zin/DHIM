import torch
import torch.nn as nn

from utils.torch_helper import move_to_device, squeeze_dim

def compute_retrieval_precision(train_loader, eval_loader, device,
                                encode_discrete=None, distance_metric='hamming',
                                num_retrieve=100):
    def extract_data(loader):
        encoding_chunks = []
        label_chunks = []
        for (docs, labels) in loader:
            docs = squeeze_dim(move_to_device(docs.data, device), dim=1)
            encoding_chunks.append(docs if encode_discrete is None else
                                   encode_discrete(docs))
            label_chunks.append(labels)

        encoding_mat = torch.cat(encoding_chunks, 0)
        label_mat = torch.cat(label_chunks, 0)
        label_lists = [[label_mat[i].item()] for i in range(label_mat.size(0))]
        return encoding_mat, label_lists
    
    src_encodings, src_label_lists = extract_data(train_loader)
    tgt_encodings, tgt_label_lists = extract_data(eval_loader)

    prec = compute_topK_average_precision(tgt_encodings, tgt_label_lists,
                                          src_encodings, src_label_lists,
                                          num_retrieve, distance_metric)
    return prec

def compute_topK_average_precision(tgt_encodings, tgt_label_lists,
                                   src_encodings, src_label_lists,
                                   num_retrieve, distance_metric='hamming',
                                   chunk_size=100, binary=True):
    K = min(num_retrieve, len(src_encodings))
    D = compute_distance(tgt_encodings, src_encodings, distance_metric,
                         chunk_size, binary)
    
    # Random here in breaking ties (e.g., may have many 0-distance neighbors),
    # but given nontrivial representations this is not an issue (hopefully).
    #
    # TODO: maybe use a stable version of topk when available,
    #   https://github.com/pytorch/pytorch/issues/27542
    _, list_topK_nearest_indices = D.topk(K, dim=1, largest=False)

    average_precision = 0.
    for i, topK_nearest_indices in enumerate(list_topK_nearest_indices):
        gold_set = set(tgt_label_lists[i])
        candidate_lists = [src_label_lists[j] for j in topK_nearest_indices]
        precision = len([_ for candidates in candidate_lists
                         if not gold_set.isdisjoint(candidates)]) / K * 100
        average_precision += precision / tgt_encodings.size(0)
    
    return average_precision

def compute_distance(X1, X2, distance_metric='hamming', chunk_size=1000,
                     binary=True):
    if distance_metric == 'hamming':
        D = compute_hamming_distance(X1, X2, chunk_size=chunk_size,
                                     binary=binary)
    elif distance_metric == 'cosine':
        D = cosine_distance_torch(X1, X2)
    else:
        raise Exception('Unsupported distance: {0}'.format(distance_metric))
    return D

def compute_hamming_distance(X1, X2, chunk_size=100, binary=True):
    assert X1.size(1) == X2.size(1)
    N, m = X1.shape
    M, m = X2.shape

    D = []
    for i in range(0, X1.size(0), chunk_size):
        X1_chunk = X1[i:i + chunk_size]
        if binary:
            A = (1 - X1_chunk).float() @ X2.t().float()  # X2 one, X1_chunk zero
            B = X1_chunk.float() @ (1 - X2).t().float()  # X1_chunk one, X2 zero
            D.append(A + B)
        else:
            n = X1_chunk.shape[0]
            # Warning: This is extremely memory-intensive.
            D.append((X1_chunk.unsqueeze(1).expand(n, M, m) != X2).sum(dim=-1))

    return torch.cat(D, dim=0)  # N x M

# Copied from https://discuss.pytorch.org/t/pairwise-cosine-distance/30961/4.
def cosine_distance_torch(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return 1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)

def compute_retrieval_precision_median_threshold(train_b, test_b, train_y, test_y, distance_metric='hamming', num_retrieve=100):
    src_encodings = train_b
    src_label_lists = train_y
    tgt_encodings = test_b
    tgt_label_lists = test_y

    prec = compute_topK_average_precision(tgt_encodings, tgt_label_lists,
                                          src_encodings, src_label_lists,
                                          num_retrieve, distance_metric)
    return prec