import argparse
import os
import itertools
import random
import sys
import numpy as np
import pickle as pk
from tqdm import tqdm
from collections import defaultdict

def preprocess(input_txt_path):
    word2id, doc_with_idx = {}, []
    with open(input_txt_path, 'r', encoding='utf-8') as f:
        for doc in f.readlines():
            for word in doc.strip().split():
                if word not in word2id:
                    word2id[word] = len(word2id)
            doc_with_idx.append([word2id[word] for word in doc.strip().split()])

    idx2word = list(word2id.keys())
    return idx2word, doc_with_idx

def mult_sample(pz_b):
    pz_b, u, k = np.cumsum(pz_b), random.random(), 0
    while (k < pz_b.shape[0]):
        if pz_b[k] >= u * pz_b[-1]:
            break
        k += 1
    if k == pz_b.shape[0]: k -= 1
    return k

def estimate(num_topics, num_words, alpha, beta, niter, doc_with_idx):

    #### get biterm_set and init
    # nz -> how many biterms are assigned to the topic z
    # nw_z -> the times of the word w assigned to the topic z
    biterm_set, nz, nw_z = [], np.zeros(num_topics), np.zeros((num_topics, num_words))
    # print(max([len(doc) for doc in doc_with_idx]))    # a window should be used if big
    for doc in doc_with_idx:
        for w1, w2 in itertools.combinations(doc, 2):  # use a window with size = 15 in BTM paper
            k = random.randint(0, num_topics - 1)
            biterm_set.append([w1, w2, k])   # k is init topic assignment
            nz[k] += 1
            nw_z[k][w1] += 1
            nw_z[k][w2] += 1

    #### Gibbs sampling
    for _ in tqdm(range(niter), desc='Gibbs sampling'):
        for biterm in biterm_set:

            # clear current assignment
            w1, w2, k = biterm[0], biterm[1], biterm[2]
            nz[k] -= 1
            nw_z[k][w1] -= 1
            nw_z[k][w2] -= 1
            # biterm[2] = -1

            # compute the conditional distribution
            p = (nz + alpha) / (len(biterm_set) + num_topics * alpha)
            pw1 = (nw_z[:, w1] + beta) / (2 * nz + num_words * beta)
            pw2 = (nw_z[:, w2] + beta) / (2 * nz + 1 + num_words * beta)
            pz_b = p * pw1 * pw2

            # assignment with new topic
            new_k = mult_sample(pz_b)   # multinominal ???
            biterm[2] = new_k
            nz[new_k] += 1
            nw_z[new_k][w1] += 1
            nw_z[new_k][w2] += 1

    #### get phi and theta
    theta = (nz + alpha) / (np.sum(nz) + num_topics * alpha)    # p(z)
    phi = (nw_z + beta) / (nz.reshape(num_topics, 1) * 2 + num_words * beta)    # p(w|z)

    return theta, phi

def inference(num_topics, doc_with_idx, theta, phi):
    pz_d = np.zeros((num_topics, len(doc_with_idx)))
    for i, doc in enumerate(doc_with_idx):
        if len(doc) == 1:   # the doc only has one word
            pz_d[:, i] = theta * phi[:, doc[0]]
        else:
            ## uniform
            # p(b|d) is uniform distribution in short text, and uniform is used in BTM c++ code
            # pz_b = np.zeros((num_topics, len(doc) * (len(doc) - 1) // 2))
            # for j, w1, w2 in enumerate(itertools.combinations(doc, 2)):
            #     pz_b[:, j] = theta * phi[:, w1] * phi[:, w2]
            # pz_d[:, i] = np.sum(pz_b / np.sum(pz_b, axis=0), axis=1)

            ## not uniform
            # (w1, w2) and (w2, w1) are same
            count_biterms = defaultdict(int)
            for w1, w2 in itertools.combinations(doc, 2):
                count_biterms[(min(w1, w2), max(w1, w2))] += 1
            pz_b, pb_d = np.zeros((num_topics, len(count_biterms))), np.zeros(len(count_biterms))
            for j, (w1, w2) in enumerate(count_biterms.keys()):
                pz_b[:, j] = theta * phi[:, w1] * phi[:, w2]
                pb_d[j] = count_biterms[(w1,w2)]
            pz_d[:, i] = np.sum((pz_b / np.sum(pz_b, axis=0)) * (pb_d / np.sum(pb_d)), axis=1)

    pz_d = pz_d / np.sum(pz_d, axis=0)
    return pz_d

def save(theta, phi, pz_d, output_dir_path):
    os.makedirs(output_dir_path, exist_ok=True)  # build output dir
    with open(os.path.join(output_dir_path, "GongJunqing.pk"), "wb") as f:
        pk.dump({
            "theta": theta,
            "phi": phi,
            "pz_d": pz_d
        }, f, protocol=4)

def BTM(args):
    # preprocess doc and get vocabulary
    idx2word, doc_with_idx = preprocess(args.input_txt_path)
    theta, phi = estimate(args.num_of_topics, len(idx2word), args.alpha, args.beta, args.niter, doc_with_idx)
    pz_d = inference(args.num_of_topics, doc_with_idx, theta, phi)
    # print(pz_d[:, 0])
    save(theta, phi, pz_d, args.output_dir_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Biterm Topic Model')
    parser.add_argument("--num-of-topics", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=2.5)     # round(50/num_of_topics, 3)
    parser.add_argument("--beta", type=float, default=0.005)
    parser.add_argument("--niter",type=int, default=3)
    parser.add_argument("--save-step", type=int, default=501)
    parser.add_argument("--input-txt-path", type=str, default="../data/doc_info.txt")
    parser.add_argument("--output-dir-path", type=str, default="../output/")
    args = parser.parse_args()

    # print(vars(args))
    BTM(args)
