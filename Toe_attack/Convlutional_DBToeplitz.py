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

    def __init__(self, input_size, kernel, stride=None, mode=None, padding=None, with_padding=True,
                 return_sparse=False):
        self.image_size = input_size
        self.channels_out = kernel.shape[0]
        self.channels_in = kernel.shape[1]
        self.kernel_size = kernel.shape[2]
        self.kernel = kernel

        self.mode = mode
        if mode is not None:
            if self.kernel_size % 2 == 0:
                raise ValueError("Kernel size should be odd.")
            if mode == "valid":
                self.padding = 0
            elif mode == "full":
                self.padding = self.kernel_size - 1
            elif mode == "same":
                self.padding = (self.kernel_size - 1) // 2
        else:
            self.padding = padding if padding is not None else 0

        if stride is not None:
            self.stride = stride

        else:
            self.stride = 1

        self.out_size = ((self.image_size - self.kernel_size + 2 * self.padding) // self.stride) + 1
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
        # 使用lil_matrix格式创建稀疏矩阵
        padding = lil_matrix((self.image_size * self.image_size, total_size))
        row_idx = 0
        for h in range(self.image_size + 2 * self.padding):
            for w in range(self.image_size + 2 * self.padding):
                if self.padding <= h < self.image_size + self.padding and self.padding <= w < self.image_size + self.padding:
                    col_idx = h * (self.image_size + 2 * self.padding) + w
                    padding[row_idx, col_idx] = 1
                    row_idx += 1

        # 将lil_matrix转换csr_matrix
        padding = padding.tocsr()
        padding = padding.transpose()

        # 使用kronecker积创建块对角矩阵
        padding = kron(identity(self.channels_in, format='csr'), padding, format='csr')

        return padding

    def creat_W2(self):
        # 初始化W2张量
        W2 = torch.zeros((self.channels_out * self.out_size * self.out_size), \
                         (self.channels_in * (self.image_size + 2 * self.padding) * (
                                     self.image_size + 2 * self.padding)))

        row_idx = 0
        for self.h in range(self.channels_out):
            for self.i in range(self.out_size):
                for self.j in range(self.out_size):

                    embedded_filter = self.psi_inverse(self.kernel[self.h], self.i * self.stride, self.j * self.stride,
                                                       self.image_size + 2 * self.padding, self.kernel_size)
                    W2[row_idx, :] = self.phi(embedded_filter)
                    row_idx += 1

        if self.padding != 0:
            padding = self.padding_operator()
            W2 = csr_matrix(W2) @ csr_matrix(padding)
            if not self.return_sparse:
                W2 = W2.todense()

        del padding
        gc.collect()

        return W2


    def get_MAX_singular_vector(self):
    #基于krylov子空间迭代法
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
 W = Convlution_as_DBToeplitz(13, W, padding=1,stride=1, return_sparse=True)
 s,vt=W.get_MAX_singular_vector()
 data_to_save = (s, vt)
 print(s)
 print(vt)
 # filename = os.path.join('/home/yyt/Project1/data/alexnet/CONV10', f'CONV10_value_vec.pth')
 # torch.save(data_to_save, filename)
 #














 # for i in range(512):
 #     print(i)
 #     # W22 = W[i].unsqueeze(0)
 #     # W22 = Convlution_as_DBToeplitz(112, W22, mode='same',return_sparse=True)
 #     # s,vt = W22.get_MAX_singular_vector()
 #     # print(s,vt.shape)
 #     # data_to_save = (s, vt)
 #     # folder_name = 'CONV10'
 #     # os.makedirs(folder_name, exist_ok=True)
 #     # filename = os.path.join(folder_name, f'CONV10_value_vec_{i}.pth')
 #     # torch.save(data_to_save, filename)
 #     # del W22,s,vt, data_to_save
 #     # gc.collect()
 #     W22 = W[i].unsqueeze(0)
 #     W22 = Convlution_as_DBToeplitz(27, W22, padding=2,stride=1, return_sparse=True)
 #     W222 = W22.creat_W2()
 #     save_npz(f'/home/yyt/Project1/data/spares_of_conv3/sparse_matrix_{i}.npz', W222)
 #     del W22,W222
 #     gc.collect()
 # #



 # W1 = torch.randn((2, 2, 5, 5))
 # X = torch.randn((1,17,17))
 # X2 = X.repeat(2, 1, 1)
 #
 # W = Convlution_as_DBToeplitz(17, W1,padding=2,stride=4, return_sparse=False)
 # W = W.creat_W2()
 # print(W.shape)
 # W = torch.tensor(W)
 # Y = torch.matmul(W.to(float),X2.view(-1).to(float))
 # print(Y.view(2,1,5,5))
 # Y_conv2d = F.conv2d(X2.unsqueeze(0), W1, padding=2,stride=(4,4))
 # print(Y_conv2d)
