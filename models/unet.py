try:
    import cupy as np
except:
    import numpy as np

import src.nn as nn

class ResBlock(nn.Layer):
    def __init__(self, in_channels, out_channels, time_embed_dim, up=False) -> None:
        self.C_in = in_channels
        self.C_out = out_channels
        self.time_embed = nn.Linear(time_embed_dim, out_channels)

        if up: #upsample
            self.conv1 = nn.Conv2D(2*in_channels, out_channels, kernel_size=(3,3), padding=1)
            self.conv_trans = nn.ConvTranspose2D(out_channels, out_channels, kernel_size=(4,4), stride=2, padding=1)
        else: # downsample
            self.conv1 = nn.Conv2D(in_channels, out_channels, kernel_size=(3,3), padding=1)
            self.conv_trans = nn.Conv2D(out_channels, out_channels, kernel_size=(4,4), stride=2, padding=1)

        self.conv2 = nn.Conv2D(out_channels, out_channels, kernel_size=(3,3), padding=1)
        self.relu1 = nn.LeakyReLU()
        self.relu2 = nn.LeakyReLU()
        self.relu3 = nn.LeakyReLU()
        
        self.batchnorm1 = nn.BatchNorm2D(out_channels)
        self.batchnorm2 = nn.BatchNorm2D(out_channels)

    def forward(self, inputs, time_embed, training=True):
        
        x = self.conv1(inputs)
        x = self.relu1(x)
        x = self.batchnorm1(x)
        
        time_embed = self.time_embed(time_embed)
        time_embed = self.relu2(time_embed)
        time_embed = time_embed[(...,) + (None, None)]
        
        x = x + time_embed
        
        x = self.conv2(x)
        x = self.relu3(x)
        x = self.batchnorm2(x)
        
        x = self.conv_trans(x)
        return x
    
    def __call__(self, inputs, time_embed, training=True, *args, **kwds):
        return self.forward(inputs, time_embed, training, *args, **kwds)
    
    def backward(self, dvalues):
        dvalues = self.conv_trans.backward(dvalues)
        
        dvalues = self.batchnorm2.backward(dvalues)
        dvalues = self.relu3.backward(dvalues)
        dvalues = self.conv2.backward(dvalues)
        
        dtime_embed = np.sum(dvalues, axis=(-2,-1))
        dtime_embed = self.relu2.backward(dtime_embed)
        dtime_embed = self.time_embed.backward(dtime_embed)
        
        dvalues = self.batchnorm1.backward(dvalues)
        dvalues = self.relu1.backward(dvalues)
        dvalues = self.conv1.backward(dvalues)
        
        return dvalues, dtime_embed
    
    @property
    def params(self):
        params = self.conv1.params + self.conv2.params + self.conv_trans.params + self.batchnorm1.params \
                + self.batchnorm2.params + self.relu1.params + self.relu2.params + self.relu3.params
        return params
    
    @property
    def grads(self):
        grads = self.conv1.grads + self.conv2.grads + self.conv_trans.grads + self.batchnorm1.grads \
                + self.batchnorm2.grads + self.relu1.grads + self.relu2.grads + self.relu3.grads
        return grads
    
    
class Unet:
    def __init__(self, image_channels, down_channels=[8,16,16], up_channels=[16,16,8], max_time=1000):
        self.image_channels = image_channels
        self.max_time = max_time
        time_embed_dim=32
        
        self.time_embed = nn.Sequential(
            nn.PositionalEncoding(max_len=max_time, output_d=time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.LeakyReLU()
        )
        
        self.input_conv = nn.Conv2D(image_channels, down_channels[0], kernel_size=(3,3), padding=1)
        self.output_conv = nn.ConvTranspose2D(up_channels[-1], image_channels, kernel_size=(3,3), padding=1)
        
        self.down_layers = [ResBlock(down_channels[i], down_channels[i+1], time_embed_dim) for i in range(len(down_channels)-1)]
        self.up_layers = [ResBlock(up_channels[i], up_channels[i+1], time_embed_dim, up=True) for i in range(len(up_channels)-1)]

        self.layers = self.up_layers + self.down_layers + [self.time_embed] +[self.output_conv, self.input_conv]
        
    def forward(self, inputs, time_steps, training=True):
        N = inputs.shape[0]
        time_steps = np.asarray(time_steps)[:, None, None]
        time_embed = self.time_embed(time_steps)
        time_embed = time_embed.reshape(N, -1)
        
        x = self.input_conv(inputs)
        
        residuals = []
        for down_layer in self.down_layers:
            x = down_layer(x, time_embed, training)
            residuals.append(x)
            
        for up_layer in self.up_layers:
            residual = residuals.pop()
            
            x = np.concatenate((x, residual), axis=1)
            x = up_layer(x, time_embed, training)
        
        outputs = self.output_conv(x)
        return outputs
    
    def __call__(self, inputs, time_steps, training=True, *args, **kwds):
        return self.forward(inputs, time_steps, training, *args, **kwds)
    
    def backward(self, dvalues):
        dvalues = self.output_conv.backward(dvalues)

        dtime_embed = 0
        dresiduals = []

        for layer in reversed(self.up_layers):
            dvalues, dtime_emb = layer.backward(dvalues)
            dtime_embed += dtime_emb
            dvalues, dresidual_x = np.split(dvalues, 2, axis=1)
            dresiduals.append(dresidual_x)

        for layer in reversed(self.down_layers):
            dresidual_x = dresiduals.pop()

            dvalues = dvalues + dresidual_x
            dvalues, dtime_emb = layer.backward(dvalues)
            dtime_embed += dtime_emb
        
        dvalues = self.input_conv.backward(dvalues)

        dtime_embed = self.time_embed.backward(dtime_embed[:,None,:])

        return dvalues
            
if __name__=='__main__':
    x = np.arange(512).reshape(2,1,16,16)
    t = np.arange(2).reshape(2)
    unet = Unet(1)
    out = unet(x, t)
    dvalue = unet.backward(out)
    print(out)