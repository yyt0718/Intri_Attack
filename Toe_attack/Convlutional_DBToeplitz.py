import gc
import torch
from scipy.sparse import csr_matrix
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse import kron, identity
from scipy.sparse.linalg import svds
import os
from scipy.sparse import csr_matrix, save_npz

class Convlution_as_DBToeplitz:

    def __init__(self,input_size,kernel,mode=None,padding=None,with_padding=True,return_sparse=False):

        self.image_size = input_size
        self.channels_out = kernel.shape[0]
        self.channels_in = kernel.shape[1]
        self.kernel_size = kernel.shape[2]
        self.kernel = kernel

        self.mode = mode
        if mode is not None:
            if self.kernel_size % 2 == 2:
                raise ValueError("kernel size should be odd.")
            if mode == "valid":
                self.padding = 0
            elif mode == "full":
                self.padding = self.kernel_size -1
            elif mode == "same":
                self.padding = (self.kernel_size - 1) // 2
        else:
            self.padding = padding


        self.out_size = (self.image_size - self.kernel_size) + 2 * self.padding + 1
        self.with_padding = with_padding
        self.return_sparse = return_sparse

    def phi(self, tensor):
        return tensor.reshape(-1)

    def phi_inverse(self, flattened_tensor, shape):
        return flattened_tensor.view(shape)

    def psi(self, X, i, j, m, k):
        return X[:, i:i + k, j:j + k]

    def psi_inverse(self, T, i, j, m, k):
        embedded_tensor = torch.zeros((self.kernel[self.h].shape[0], m, m))
        embedded_tensor[:, i:i + k, j:j + k] = T
        return embedded_tensor

    def padding_operator(self):
        total_size = (self.image_size + 2 * self.padding) * (self.image_size + 2 * self.padding)
        # 稀疏矩阵
        padding = lil_matrix((self.image_size * self.image_size, total_size))
        row_idx = 0
        for h in range(self.image_size + 2 * self.padding):
            for w in range(self.image_size + 2 * self.padding):
                if self.padding <= h < self.image_size + self.padding and self.padding <= w < self.image_size + self.padding:
                    col_idx = h * (self.image_size + 2 * self.padding) + w
                    padding[row_idx, col_idx] = 1
                    row_idx += 1

        # lil_matrix to csr_matrix
        padding = padding.tocsr()
        padding = padding.transpose()

        # kronecker
        padding = kron(identity(self.channels_in, format='csr'), padding, format='csr')

        return padding

    def creat_W2(self):


        W2 = torch.zeros((self.channels_out * self.out_size * self.out_size), \
                         (self.channels_in * (self.image_size + 2 * self.padding) * (self.image_size + 2 * self.padding)))

        for self.h in range(self.channels_out):
            for self.i in range(self.out_size):
                for self.j in range(self.out_size):
                    row_idx = self.h * self.out_size * self.out_size + self.i * self.out_size + self.j
                    embedded_filter = self.psi_inverse(self.kernel[self.h], self.i, self.j, self.image_size + 2 * self.padding, self.kernel_size)
                    W2[row_idx, :] = self.phi(embedded_filter)

        if self.padding != 0:
            padding = self.padding_operator()
            W2 = csr_matrix(W2) @ csr_matrix(padding)
            if not self.return_sparse:
                W2 = W2.todense()

        del padding
        gc.collect()

        return W2


    def get_MAX_singular_vector(self):

        W2 = self.creat_W2()
        k = 1
        _, s, vt = svds(W2, k=k)
        vt = vt.copy()
        vt = torch.from_numpy(vt)

        s = s.copy()
        s = torch.from_numpy(s)

        return s , vt




if __name__ == "__main__":

 import torch
 import torch.nn.functional as F
 from torchvision.models import alexnet

 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 model = alexnet(pretrained=True).to(device)
 model.eval()
 W = model.features[10].weight
 W = W.detach()


 for i in range(512):
     print(i)
     # W22 = W[i].unsqueeze(0)
     # W22 = Convlution_as_DBToeplitz(112, W22, mode='same',return_sparse=True)
     # s,vt = W22.get_MAX_singular_vector()
     # print(s,vt.shape)
     # data_to_save = (s, vt)
     # folder_name = 'CONV10'
     # os.makedirs(folder_name, exist_ok=True)
     # filename = os.path.join(folder_name, f'CONV10_value_vec_{i}.pth')
     # torch.save(data_to_save, filename)
     # del W22,s,vt, data_to_save
     # gc.collect()
     W22 = W[i].unsqueeze(0)
     W22 = Convlution_as_DBToeplitz(56, W22, mode='same', return_sparse=True)
     W222 = W22.creat_W2()
     save_npz(f'/home/yyt/Project1/data/spares_of_conv10/sparse_matrix_{i}.npz', W222)
     del W22,W222
     gc.collect()

# W22 = Convlution_as_DBToeplitz(224, W, mode='same',return_sparse=True)
# s,vt = W22.get_MAX_singular_vector()
# print(s,vt.shape)
# data_to_save = (s, vt)
# folder_name = '/home/yyt/Project1/data/vgg19/CONV34'
# os.makedirs(folder_name, exist_ok=True)
# filename = os.path.join(folder_name, f'CONV34_value_vec.pth')
# torch.save(data_to_save, filename)