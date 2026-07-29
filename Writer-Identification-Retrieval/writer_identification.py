import os
import shlex
import argparse
from multiprocessing.pool import ThreadPool

from sklearn.decomposition import PCA
from tqdm import tqdm

# for python3: read in python2 pickled files
import _pickle as cPickle

import gzip
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import LinearSVC
from sklearn.linear_model import Ridge
from sklearn.preprocessing import normalize
import numpy as np
import cv2
from parmap import parmap

import warnings
warnings.filterwarnings("ignore")

def parseArgs(parser):
    parser.add_argument('--labels_test',
                        default='./icdar17_local_features/icdar17_labels_test.txt',
                        help='contains test images/descriptors to load + labels')
    parser.add_argument('--labels_train',
                        default='./icdar17_local_features/icdar17_labels_train.txt',
                        help='contains training images/descriptors to load + labels')
    parser.add_argument('-s', '--suffix',
                        default='_SIFT_patch_pr.pkl.gz',
                        help='only chose those images with a specific suffix')
    parser.add_argument('--suffix_train', default='.png',
                        help='only chose those images with a specific suffix')
    parser.add_argument('--suffix_test', default='.jpg',
                        help='only chose those images with a specific suffix')
    parser.add_argument('--in_test',
                        # default='./icdar17_local_features/test',
                        default='./data/testset/ScriptNet-HistoricalWI-2017-binarized',
                        help='the input folder of the test images / features')
    parser.add_argument('--in_train',
                        # default='./icdar17_local_features/train',
                        default='./data/trainset/icdar2017-training-binary',
                        help='the input folder of the training images / features')
    parser.add_argument('--overwrite', action='store_true',
                        help='do not load pre-computed encodings')
    parser.add_argument('--powernorm', action='store_true',
                        help='use powernorm')
    parser.add_argument('--gmp', action='store_true',
                        help='use generalized max pooling')
    parser.add_argument('--gamma', default=1, type=float,
                        help='regularization parameter of GMP')
    parser.add_argument('--C', default=1000, type=float, 
                        help='C parameter of the SVM')
    parser.add_argument('--multivlad', action='store_true',
                        help='use multi vlad')
    return parser


def getFiles(folder, pattern, labelfile):
    """ 
    returns files and associated labels by reading the labelfile 
    parameters:
        folder: inputfolder
        pattern: new suffix
        labelfiles: contains a list of filename and labels
    return: absolute filenames + labels 
    """
    # read labelfile
    with open(labelfile, 'r') as f:
        all_lines = f.readlines()
    
    # get filenames from labelfile
    all_files = []
    labels = []
    check = True
    for line in all_lines:
        # using shlex we also allow spaces in filenames when escaped w. ""
        splits = shlex.split(line)
        file_name = splits[0]
        class_id = splits[1]

        # strip all known endings, note: os.path.splitext() doesnt work for
        # '.' in the filenames, so let's do it this way...
        for p in ['.pkl.gz', '.txt', '.png', '.jpg', '.tif', '.ocvmb','.csv']:
            if file_name.endswith(p):
                file_name = file_name.replace(p,'')

        # get now new file name
        true_file_name = os.path.join(folder, file_name + pattern)
        all_files.append(true_file_name)
        labels.append(class_id)

    return all_files, labels

def loadRandomDescriptors(files, max_descriptors):
    """
    load roughly `max_descriptors` random descriptors
    parameters:
        files: list of filenames containing local features of dimension D
        max_descriptors: maximum number of descriptors (Q)
    returns: QxD matrix of descriptors
    """
    # let's just take 100 files to speed-up the process
    max_files = 100
    indices = np.random.permutation(max_files)
    files = np.array(files)[indices]

    # rough number of descriptors per file that we have to load
    max_descs_per_file = int(max_descriptors / len(files))

    descriptors = []
    for i in tqdm(range(len(files))):
        # with gzip.open(files[i], 'rb') as ff:
            # for python2
            # desc = cPickle.load(ff)
            # for python3
            # desc = cPickle.load(ff, encoding='latin1')
        desc = computeDescs(files[i])

        # get some random ones
        indices = np.random.choice(len(desc),
                                   min(len(desc),
                                       int(max_descs_per_file)),
                                   replace=False)
        desc = desc[ indices ]
        descriptors.append(desc)
    
    descriptors = np.concatenate(descriptors, axis=0)
    return descriptors

def dictionary(descriptors, n_clusters, random_state=42):
    """ 
    return cluster centers for the descriptors 
    parameters:
        descriptors: NxD matrix of local descriptors
        n_clusters: number of clusters = K
    returns: KxD matrix of K clusters
    """

    k_means = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=1000,
        random_state=random_state
    )
    k_means.fit(descriptors)
    mus = k_means.cluster_centers_
    return mus

def assignments(descriptors, clusters):
    """ 
    compute assignment matrix
    parameters:
        descriptors: TxD descriptor matrix
        clusters: KxD cluster matrix
    returns: TxK assignment matrix
    """

    # compute nearest neighbors
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    matches = matcher.knnMatch(descriptors, clusters, k=1)

    # create hard assignment
    assignment = np.zeros((len(descriptors), len(clusters)))

    for i, match in enumerate(matches):
        # get the index of first element which is the nearest cluster
        idx = match[0].trainIdx
        assignment[i, idx] = 1.0

    return assignment

def vlad(files, mus, powernorm, gmp=False, gamma=1000):
    """
    compute VLAD encoding for each files
    parameters: 
        files: list of N files containing each T local descriptors of dimension
        D
        mus: KxD matrix of cluster centers
        gmp: if set to True use generalized max pooling instead of sum pooling
    returns: NxK*D matrix of encodings
    """
    K = mus.shape[0]

    # here we can use this loop function to do multi threading and increase the speed because it was running too slowly.
    def loop(f):
        desc = computeDescs(f)
        a = assignments(desc, mus)

        T, D = desc.shape
        f_enc = np.zeros((D*K), dtype=np.float32)

        for k in range(K):
            idx = np.where(a[:, k] == 1)[0]
            if len(idx) == 0:
                continue

            residuals = desc[idx] - mus[k]

            if gmp:
                # using generalized max pooling by Ridge function instead of normal summation of residuals
                y = np.ones(len(idx))
                ridge = Ridge( alpha=gamma, solver='sparse_cg', fit_intercept=False, max_iter=500)
                ridge.fit(residuals, y)
                res_aggregated = ridge.coef_
            else:
                # use normal summation
                res_aggregated = np.sum(residuals, axis=0)


            # insert the result into f_enc
            f_enc[k * D:(k + 1) * D] = res_aggregated

        # c) power normalization
        if powernorm:
            f_enc = np.sign(f_enc) * np.sqrt(np.abs(f_enc))

        # l2 normalization
        norm = np.linalg.norm(f_enc)
        if norm != 0:
            f_enc /= norm

        return f_enc

    with ThreadPool() as pool:
        encodings = list(tqdm(pool.imap(loop, files), total=len(files)))

    return np.array(encodings)


def multiVlad(files, mus_list, powernorm, gmp=False, gamma=1000):
    """
    compute multi VLAD encoding for each files
    parameters:
        files: list of N files containing each T local descriptors of dimension
        D
        mus_list: list of 5 mus of cluster centers
        gmp: if set to True use generalized max pooling instead of sum pooling
    returns: NxK*D matrix of encodings
    """
    K = mus_list[0].shape[0]

    # here we can use this loop function to do multi threading and increase the speed because it was running to slowly.
    def loop(f):
        desc = computeDescs(f)
        T, D = desc.shape

        enc_list = []
        for mus in mus_list:
            a = assignments(desc, mus)
            f_enc = np.zeros((D*K), dtype=np.float32)

            for k in range(K):
                idx = np.where(a[:, k] == 1)[0]
                if len(idx) == 0:
                    continue

                residuals = desc[idx] - mus[k]

                if gmp:
                    # using generalized max pooling by Ridge function instead of normal summation of residuals
                    y = np.ones(len(idx))
                    ridge = Ridge( alpha=gamma, solver='sparse_cg', fit_intercept=False, max_iter=500)
                    ridge.fit(residuals, y)
                    res_aggregated = ridge.coef_
                else:
                    # use normal summation
                    res_aggregated = np.sum(residuals, axis=0)

                # insert the result into f_enc
                f_enc[k * D:(k + 1) * D] = res_aggregated

            # c) power normalization
            if powernorm:
                f_enc = np.sign(f_enc) * np.sqrt(np.abs(f_enc))

            # l2 normalization
            norm = np.linalg.norm(f_enc)
            if norm != 0:
                f_enc /= norm

            enc_list.append(f_enc)

        f_enc = np.concatenate(enc_list)
        return f_enc

    with ThreadPool() as pool:
        encodings = list(tqdm(pool.imap(loop, files), total=len(files)))

    return np.array(encodings)


def esvm(encs_test, encs_train, C=1000):
    """ 
    compute a new embedding using Exemplar Classification
    compute for each encs_test encoding an E-SVM using the
    encs_train as negatives   
    parameters: 
        encs_test: NxD matrix
        encs_train: MxD matrix

    returns: new encs_test matrix (NxD)
    """

    # set up labels
    y = np.zeros(len(encs_train) + 1)
    # Set the first element as the Positive class
    y[0] = 1

    def loop(i):
        # compute SVM
        # and make feature transformation

        # set up input values, the first should be the test enc with positive class
        X = np.zeros((len(encs_train) + 1, encs_train.shape[1]))
        X[0] = encs_test[i]
        X[1:] = encs_train

        # compute svm
        svm = LinearSVC(C=C, class_weight='balanced', random_state=42)
        svm.fit(X, y)
        weights = svm.coef_[0]

        # normalize weights
        norm = np.linalg.norm(weights)
        if norm > 0:
            weights /=  norm
        weights = weights.reshape(1, -1)

        return weights

    # let's do that in parallel:
    # if that doesn't work for you, just exchange 'parmap' with 'map'
    # Even better: use DASK arrays instead, then everything should be
    # parallelized
    # new_encs = list(parmap(loop, tqdm(range(len(encs_test)))))

    #parmap did not work worked for me so I changed it to use multi threading instead to speed up
    with ThreadPool() as pool:
        new_encs = list(tqdm(pool.imap(loop, range(len(encs_test))), total=len(encs_test)))

    new_encs = np.concatenate(new_encs, axis=0)
    # return new encodings
    return new_encs


def distances(encs):
    """ 
    compute pairwise distances 

    parameters:
        encs:  TxK*D encoding matrix
    returns: TxT distance matrix
    """
    # compute cosine distance = 1 - dot product between l2-normalized
    # encodings
    # compute norm to ensure that we have the l2 normalized encs
    norm = np.linalg.norm(encs, axis=1, keepdims=True)
    norm[norm == 0] = 1 # to avoid division by 0
    encs = encs / norm

    # compute distances of l2 normalized
    dists = 1 - np.dot(encs, encs.T)

    # mask out distance with itself
    np.fill_diagonal(dists, np.finfo(dists.dtype).max)
    return dists

def evaluate(encs, labels):
    """
    evaluate encodings assuming using associated labels
    parameters:
        encs: TxK*D encoding matrix
        labels: array/list of T labels
    """
    dist_matrix = distances(encs)
    # sort each row of the distance matrix
    indices = dist_matrix.argsort()

    n_encs = len(encs)

    mAP = []
    correct = 0
    for r in range(n_encs):
        precisions = []
        rel = 0
        for k in range(n_encs-1):
            if labels[ indices[r,k] ] == labels[ r ]:
                rel += 1
                precisions.append( rel / float(k+1) )
                if k == 0:
                    correct += 1
        avg_precision = np.mean(precisions)
        mAP.append(avg_precision)
    mAP = np.mean(mAP)

    print('Top-1 accuracy: {} - mAP: {}'.format(float(correct) / n_encs, mAP))


def computeDescs(filename):
    """
    This function helps us to extract our own features instead of pre-computated ones.
    """

    # read the image
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    # compute SIFT
    sift = cv2.SIFT_create()
    keypoints = sift.detect(img, None)

    # change angles of the keypoints to 0
    for keypoint in keypoints:
        keypoint.angle = 0.0

    _, desc = sift.compute(img, keypoints)
    if desc is None:
        return np.zeros((0, 128), dtype=np.float32)

    # apply Hellinger normalization
    desc = normalize(desc, norm='l1', axis=1)
    desc = np.sign(desc) * np.sqrt(np.abs(desc))

    return desc



if __name__ == '__main__':
    parser = argparse.ArgumentParser('retrieval')
    parser = parseArgs(parser)
    args = parser.parse_args()
    np.random.seed(42) # fix random seed

    files_train, labels_train = getFiles(args.in_train, args.suffix_train,
                                         args.labels_train)
    print('#train: {}'.format(len(files_train)))

    files_test, labels_test = getFiles(args.in_test, args.suffix_test,
                                       args.labels_test)
    print('#test: {}'.format(len(files_test)))

    # multi vlad
    if args.multivlad:
        # calculating or reading 5 dictionaries
        if not os.path.exists('multi_mus.pkl.gz') or args.overwrite:

            # cluster centers
            print('> compute multi dictionaries')
            mus_list = []
            for i in range(5):
                descriptors = loadRandomDescriptors(files_train, max_descriptors=500000)
                mus = dictionary(descriptors, n_clusters=100, random_state=42+i)
                mus_list.append(mus)

            with gzip.open('multi_mus.pkl.gz', 'wb') as fOut:
                cPickle.dump(mus_list, fOut, -1)

        else:
            with gzip.open('multi_mus.pkl.gz', 'rb') as f:
                mus_list = cPickle.load(f)

        # VLAD encoding
        print('> compute multi VLAD for train and test')

        fname_train = 'multi_enc_train_gmp{}.pkl.gz'.format(args.gamma) if args.gmp else 'multi_enc_train.pkl.gz'
        fname_test = 'multi_enc_test_gmp{}.pkl.gz'.format(args.gamma) if args.gmp else 'multi_enc_test.pkl.gz'

        if not os.path.exists(fname_train) or not os.path.exists(fname_test) or args.overwrite:

            enc_train = multiVlad(files_train, mus_list, powernorm=args.powernorm,
                                  gmp=args.gmp, gamma=args.gamma)
            enc_test = multiVlad(files_test, mus_list, powernorm=args.powernorm,
                                 gmp=args.gmp, gamma=args.gamma)
            # print("enc_train shape:", enc_train.shape, "enc_test shape:", enc_test.shape)

            # applying PCA to reduce dimensionality
            pca = PCA(n_components=1000, whiten=True, random_state=42)
            pca.fit(enc_train)

            enc_train = pca.transform(enc_train)
            enc_test = pca.transform(enc_test)

            # l2 normalize encs
            enc_train = normalize(enc_train, norm='l2', axis=1)
            enc_test = normalize(enc_test, norm='l2', axis=1)

            # save encs
            with gzip.open(fname_train, 'wb') as fOut:
                cPickle.dump(enc_train, fOut, -1)

            with gzip.open(fname_test, 'wb') as fOut:
                cPickle.dump(enc_test, fOut, -1)
        else:
            with gzip.open(fname_train, 'rb') as f:
                enc_train = cPickle.load(f)

            with gzip.open(fname_test, 'rb') as f:
                enc_test = cPickle.load(f)

    # normal vlad
    else:
        # a) dictionary
        if not os.path.exists('mus.pkl.gz'):
            descriptors = loadRandomDescriptors(files_train, max_descriptors=500000)
            print('> loaded {} descriptors:'.format(len(descriptors)))

            # cluster centers
            print('> compute dictionary')
            mus = dictionary(descriptors, n_clusters=100)

            with gzip.open('mus.pkl.gz', 'wb') as fOut:
                cPickle.dump(mus, fOut, -1)
        else:
            with gzip.open('mus.pkl.gz', 'rb') as f:
                mus = cPickle.load(f)

        # b) VLAD encoding
        print('> compute VLAD for train')
        fname = 'enc_train_gmp{}.pkl.gz'.format(args.gamma) if args.gmp else 'enc_train.pkl.gz'
        if not os.path.exists(fname) or args.overwrite:
            enc_train = vlad(files_train, mus,
                             powernorm=args.powernorm,
                             gmp=args.gmp,
                             gamma=args.gamma)
            with gzip.open(fname, 'wb') as fOut:
                cPickle.dump(enc_train, fOut, -1)
        else:
            with gzip.open(fname, 'rb') as f:
                enc_train = cPickle.load(f)

        print('> compute VLAD for test')

        fname = 'enc_test_gmp{}.pkl.gz'.format(args.gamma) if args.gmp else 'enc_test.pkl.gz'
        if not os.path.exists(fname) or args.overwrite:
            enc_test = vlad(files_test, mus,
                            powernorm=args.powernorm,
                            gmp=args.gmp,
                            gamma=args.gamma)

            with gzip.open(fname, 'wb') as fOut:
                cPickle.dump(enc_test, fOut, -1)
        else:
            with gzip.open(fname, 'rb') as f:
                enc_test = cPickle.load(f)


    # cross-evaluate test encodings
    print('> evaluate')
    evaluate(enc_test, labels_test)

    # d) compute exemplar svms
    print('> esvm computation')
    enc_test_svm = esvm(enc_test, enc_train, C=args.C)

    # eval
    evaluate(enc_test_svm, labels_test)
    print('> evaluate')