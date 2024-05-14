try:
    import cupy as np
except:
    import numpy as np
import src.nn as nn

class ResBlock(nn.Layer):
    def __init__(self, in_channels, out_channels, stride=1) -> None:
        self.C_in = in_channels
        self.C_out = out_channels
        self.stride = stride
        
        self.convs = nn.Sequential(
            nn.Conv2D(in_channels, out_channels, kernel_size=(3,3), stride=stride, padding=1, use_bias=False),
            nn.BatchNorm2D(out_channels),
            nn.ReLU(),
            nn.Conv2D(out_channels, out_channels, kernel_size=(3,3), stride=1, padding=1, use_bias=False),
            nn.BatchNorm2D(out_channels)
        )
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2D(in_channels, out_channels, kernel_size=(1,1), stride=stride, use_bias=False),
                nn.BatchNorm2D(out_channels)
            )
        self.relu = nn.ReLU()
        
    def forward(self, x, training=True):
        out = self.convs(x)
        out = out + self.shortcut(x)
        out = self.relu(out)
        return out
    
    def backward(self, dvalues):
        dvalues = self.relu.backward(dvalues)
        dconvs = self.convs.backward(dvalues)
        dshortcut = self.shortcut.backward(dvalues)
        dinputs = dconvs + dshortcut
        return dinputs
    

class ResNet18(nn.Layer):
    def __init__(self, image_channels, num_classes=10) -> None:
        self.num_classes = num_classes
        self.image_channels = image_channels
        
        self.C_in = 64
        self.conv1 = nn.Sequential(
            nn.Conv2D(3, 64, kernel_size=(3,3), stride=1, padding=1, use_bias=False),
            nn.BatchNorm2D(64),
            nn.ReLU()
        )
        
        self.layer1 = self.make_layer(ResBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResBlock, 256, 2, stride=2)        
        self.layer4 = self.make_layer(ResBlock, 512, 2, stride=2)
        self.avg_pool = nn.AvgPooling(4, 4, 4)
        self.flatten = nn.Flatten()        
        self.fc = nn.Linear(512, num_classes)
        self.softmax = nn.SoftMax()
        
        self.layers = [self.conv1,self.layer1,self.layer2,self.layer3,self.layer4,self.avg_pool,self.flatten,self.fc]
        
    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks-1)
        layers = []
        
        for stride in strides:
            layers.append(block(self.C_in, channels, stride))
            self.C_in = channels
            
        return nn.Sequential(*layers)
    
    def forward(self, x, training=True):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = self.flatten(out)
        out = self.fc(out)
        out = self.softmax(out)
        return out
    
    def backward(self, dvalues):
        dvalues = self.softmax.backward(dvalues)
        dvalues = self.fc.backward(dvalues)
        dvalues = self.flatten.backward(dvalues)
        dvalues = self.avg_pool.backward(dvalues)
        dvalues = self.layer4.backward(dvalues)
        dvalues = self.layer3.backward(dvalues)
        dvalues = self.layer2.backward(dvalues)
        dvalues = self.layer1.backward(dvalues)
        dvalues = self.conv1.backward(dvalues)
        return dvalues
    
    