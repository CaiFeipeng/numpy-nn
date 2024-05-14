import numpy as np
import src.nn as nn

class ResBlock(nn.Layer):
    def __init__(self, in_channels, out_channels, up=False) -> None:
        self.C_in = in_channels
        self.C_out = out_channels

        if up: #upsample
            self.conv1 = nn.Conv2D(in_channels, out_channels, kernel_size=(3,3), padding=1)
            self.conv_trans = nn.ConvTranspose2D(out_channels, out_channels, kernel_size=(4,4), stride=2, padding=1)
        else: # downsample
            self.conv1 = nn.Conv2D(in_channels, out_channels, kernel_size=(3,3), padding=1)
            self.conv_trans = nn.Conv2D(out_channels, out_channels, kernel_size=(4,4), stride=2, padding=1)

        self.conv2 = nn.Conv2D(out_channels, out_channels, kernel_size=(3,3), padding=1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        
        self.batchnorm1 = nn.BatchNorm2D(out_channels)
        self.batchnorm2 = nn.BatchNorm2D(out_channels)

    def forward(self, inputs, training=True):
        
        x = self.conv1(inputs)
        x = self.batchnorm1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu2(x)
        
        x = self.conv_trans(x)
        return x
    
    def __call__(self, inputs, training=True, *args, **kwds):
        return self.forward(inputs, training, *args, **kwds)
    
    def backward(self, dvalues):
        dvalues = self.conv_trans.backward(dvalues)
        
        dvalues = self.relu2.backward(dvalues)
        dvalues = self.batchnorm2.backward(dvalues)
        dvalues = self.conv2.backward(dvalues)
        
        dvalues = self.relu1.backward(dvalues)
        dvalues = self.batchnorm1.backward(dvalues)
        dvalues = self.conv1.backward(dvalues)
        
        return dvalues
    
    @property
    def params(self):
        params = self.conv1.params + self.conv2.params + self.conv_trans.params + self.batchnorm1.params \
                + self.batchnorm2.params + self.relu1.params + self.relu2.params
        return params
    
    @property
    def grads(self):
        grads = self.conv1.grads + self.conv2.grads + self.conv_trans.grads + self.batchnorm1.grads \
                + self.batchnorm2.grads + self.relu1.grads + self.relu2.grads
        return grads
   
class Autoencoder(nn.Layer):
    
    def __init__(self, image_channels, down_channels=[8,16,16], up_channels=[16,16,8]):
        self.image_channels = image_channels
        
        self.input_conv = nn.Conv2D(image_channels, down_channels[0], kernel_size=(3,3), padding=1)
        self.output_conv = nn.ConvTranspose2D(up_channels[-1], image_channels, kernel_size=(3,3), padding=1)
        
        encoder = [self.input_conv] + [ResBlock(down_channels[i], down_channels[i+1]) \
                                       for i in range(len(down_channels)-1)]
        self.encoder = nn.Sequential(*encoder)
        decoder = [ResBlock(up_channels[i], up_channels[i+1], up=True) for i in range(len(up_channels)-1)] \
                    + [self.output_conv]
        self.decoder = nn.Sequential(*decoder)
        self.layers = [self.decoder, self.encoder]
        
        self.mse_loss = nn.MSELoss()
        
    def forward(self, inputs, training=True):

        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded
    
    def __call__(self, inputs, training=True, *args, **kwds):
        return self.forward(inputs, training, *args, **kwds)
    
    def backward(self, dvalues):
        dvalues = self.decoder.backward(dvalues)
        dvalues = self.encoder.backward(dvalues)
        return dvalues
    @property
    def params(self):
        params = self.decoder.params + self.encoder.params
        return params
    @property
    def grads(self):
        grads = self.decoder.params + self.encoder.params
        return grads
    
    def loss_func(self, preds, labels):

        return self.mse_loss(preds, labels)
        
class VAE(Autoencoder):
    def __init__(self, image_channels, down_channels=[8, 16, 16], up_channels=[16, 16, 8]):
        super().__init__(image_channels, down_channels, up_channels)
        self.mu_layer = nn.Conv2D(down_channels[-1], down_channels[-1], kernel_size=(1,1))
        self.log_var_layer = nn.Conv2D(down_channels[-1], down_channels[-1], kernel_size=(1,1))
        self.layers = [self.decoder, self.encoder, self.mu_layer, self.log_var_layer]

    def reparameterize(self, mu, log_var):
        self. eps = np.random.randn(*mu.shape)
        std = np.exp(0.5* log_var)
        z = mu + std * self.eps
        return z
    
    def reparameterize_backward(self,dvalues):
        dmu = dvalues
        dstd = dvalues * self.eps
        dlog_var = dstd * 0.5* np.exp(0.5*self.log_var)
        return dmu, dlog_var
    
    def forward(self, inputs, training=True):
        encoded = self.encoder(inputs)
        
        self.mu = self.mu_layer(encoded)
        self.log_var = self.log_var_layer(encoded)

        z = self.reparameterize(self.mu, self.log_var)
        decoded = self.decoder(z)
        return encoded, decoded
    
    def __call__(self, inputs, training=True, *args, **kwds):
        return self.forward(inputs, training, *args, **kwds)
    
    def backward(self, dvalues, dmu_kld, dlogvar_kld):
        dvalues = self.decoder.backward(dvalues)

        dmu, dlog_var = self.reparameterize_backward(dvalues)
        dmu = dmu + dmu_kld
        dmu = self.mu_layer.backward(dmu)
        dlog_var = dlog_var + dlogvar_kld
        dlog_var = self.log_var_layer.backward(dlog_var)

        dencoded =dmu + dlog_var
        dinputs = self.encoder.backward(dencoded)
        return dinputs
    @property
    def params(self):
        params = self.decoder.params + self.encoder.params + self.mu_layer.params + self.log_var_layer.params
        return params
    @property
    def grads(self):
        grads = self.decoder.grads + self.encoder.grads + self.mu_layer.grads + self.log_var_layer.grads
        return grads
    
    def loss_func(self, preds, labels):
        drecons, loss_recons = self.mse_loss(preds, labels)

        # Divergience KL Loss
        loss_kld = -0.5 * np.sum(1 + self.log_var - self.mu**2 - np.exp(self.log_var))
        # Derivatives of DKL
        dmu = self.mu
        dlog_var = -0.5 * (1 - np.exp(self.log_var))
        return drecons, dmu, dlog_var, loss_recons + loss_kld
    
    
    
