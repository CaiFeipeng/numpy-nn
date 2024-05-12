import numpy as np

def get_im2col_indices(x_shape, kernel_height, kernel_width, padding=0, stride=1):
    N,C_in,H_in,W_in = x_shape
    
    H_out = (H_in + 2*padding - kernel_height) // stride + 1
    W_out = (W_in + 2*padding - kernel_width) // stride + 1
    
    # compute indices of i [C_in * w_k * h_k, W_out * H_out] (H_dim)
    i0 = np.repeat(np.arange(kernel_width), kernel_height)
    i1 = np.repeat(np.arange(H_out) , W_out) * stride
    i = np.tile(i0, C_in).reshape(-1, 1) + i1.reshape(1, -1)
    
    # compute indices of j (W_dim)
    j0 = np.tile(np.arange(kernel_width), kernel_height)
    j1 = np.tile(np.arange(W_out), H_out) * stride
    j = np.tile(j0, C_in).reshape(-1, 1) + j1.reshape(1, -1)
    
    # compute indices of k (C_dim)
    k = np.repeat(np.arange(C_in), kernel_width * kernel_height).reshape(-1,1)
    return k, i, j

def im2col(x, kernel_height, kernel_width, padding=0, stride=1):
    # x: [batch_size, C_in, H_in, W_in]
    x_padded = np.pad(x, ((0,0), (0,0), (padding,padding), (padding,padding)), mode='constant')
    
    k, i, j = get_im2col_indices(x.shape, kernel_height, kernel_width, padding, stride)
    
    # cols: [batch_size, C_in * kernel_height * kernel_width, H_out*W_out]
    cols = x_padded[:, k, i, j]
    return cols

def col2im(cols, array_shape, kernel_height, kernel_width, padding=0, stride=1):
    # cols: [batch_size, C_in * kernel_height * kernel_width, H_out * W_out]
    N, C_in, H_in, W_in = array_shape
    H_padded, W_padded = H_in + 2*padding, W_in + 2*padding
    
    array_padded = np.zeros((N, C_in, H_padded, W_padded))
    
    k, i, j = get_im2col_indices(array_shape, kernel_height, kernel_width, padding, stride)
    
    np.add.at(array_padded, (slice(None), k, i, j), cols)
    return array_padded[:,:,padding:padding+H_in, padding:padding+W_in]
    
if __name__=='__main__':
    x = np.arange(96).reshape(2, 3, 4, 4)
    cols = im2col(x, 3, 3, padding=1)
    
    out = col2im(cols, x.shape, 3, 3, padding=1)