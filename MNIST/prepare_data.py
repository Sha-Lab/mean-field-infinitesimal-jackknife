from torchvision import datasets
import numpy as np
'''
Split MNIST dataset into train and heldout fold, 
and save results to data/ dir.

Split NotMnist dataset into heldout and test fold,
and save results to data/ dir
'''

def prepare_mnist():
    train_set = datasets.MNIST('data/', train=True, download=True)
    test_set = datasets.MNIST('data/', train=False, download=True)

    train_data_ = train_set.data.numpy()
    train_labels_ = train_set.targets.numpy()
    test_data = test_set.data.numpy()
    test_labels = test_set.targets.numpy()

    train_data_ = train_data_ / np.float32(255)
    train_labels_ = train_labels_.astype(np.int32)

    test_data = test_data / np.float32(255)
    test_labels = test_labels.astype(np.int32)

    np.random.seed(1)
    num_train = len(train_labels_)
    index = np.random.permutation(num_train)

    num_heldout = 5000
    heldout_data = train_data_[index[:num_heldout]]
    heldout_labels = train_labels_[index[:num_heldout]]

    train_data = train_data_[index[num_heldout:]]
    train_labels = train_labels_[index[num_heldout:]]
    np.savez("data/mnist_split.npz",
             train_data=train_data, train_labels=train_labels,
             heldout_data=heldout_data, heldout_labels=heldout_labels,
             test_data=test_data, test_labels=test_labels)

def prepare_notmnist():
    np.random.seed(0)
    # notmnist is downloaded from http://yaroslavvb.com/upload/notMNIST/notMNIST_small.tar.gz
    # and saved into npz file.
    dataset = np.load('data/data_notMNIST_small.npz')
    heldout_idx = []
    test_idx = []
    num = len(dataset['labels'])
    print('there are', num, 'samples!')
    ridx = np.random.permutation(num)
    labels = dataset['labels'][ridx]
    for c in range(10):
        c_idx = ridx[np.where(labels == c)[0]]
        heldout_idx.extend(c_idx[:500])
        test_idx.extend(c_idx[500:])
    np.savez('data/notmnist_split_idx.npz', heldout_idx=heldout_idx, test_idx=test_idx)

def main():
    prepare_mnist()
    prepare_notmnist()

if __name__ == '__main__':
    main()