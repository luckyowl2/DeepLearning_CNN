%function test_cnn_gradients_are_numerically_correct
clear all
clc
batch_x = rand(28,28,5);
batch_y = rand(10,5);
cnn.layers = {
    struct('type', 'i') %input layer
    struct('type', 'c', 'outputmaps', 2, 'kernelsize', 5, 'activation', 'sigm') %convolution layer
    struct('type', 's', 'scale', 2, 'method', 'a') %sub sampling layer
    struct('type', 'c', 'outputmaps', 2, 'kernelsize', 5, 'activation', 'sigm') %convolution layer
    struct('type', 's', 'scale', 2, 'method', 'a') %subsampling layer
    struct('type','o','objective','sigm');
};
cnn.fc={
    %struct('type','f','numsoutput',30, 'activation', 'sigm','dropoutFraction',0,'dropconnectFraction',0)
    %struct('type','f','numsoutput',20, 'activation', 'sigm','dropoutFraction',0,'dropconnectFraction',0)
    %struct('type','f','numsoutput',100)
    %struct('type','f','numsoutput',30)
};
opts.momSwitch=false;
cnn.testing=false;
cnn = cnnsetup(cnn, batch_x, batch_y);

cnn = cnnff(cnn, batch_x);
cnn = cnnbp(cnn, batch_y);
cnnnumgradcheck(cnn, batch_x, batch_y);