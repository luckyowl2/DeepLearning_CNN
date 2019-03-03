% function test_example_CNN
clear all;
close all;
clc;
%%
global useGpu;
useGpu = false;%GPU使用�?��，此处尚�?���?
%%
load mnist_uint8;%手写数字样本测试代码
train_x = double(reshape(train_x',28,28,60000))/255;
test_x = double(reshape(test_x',28,28,10000))/255;
train_y = double(train_y');
test_y = double(test_y');

%% ex1 Train a 6c-2s-12c-2s Convolutional neural network 
%will run 1 epoch in about 200 second and get around 11% error. 
%With 100 epochs you'll get around 1.2% error
%%
%MaxPooling  C++代码编译
if exist('MaxPooling')~=3
   % mex MaxPooling.cpp COMPFLAGS="/openmp $COMPFLAGS" CXXFLAGS="\$CFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp" -largeArrayDims 
end;
%StochasticPooling C++代码编译
if exist('StochasticPooling')~=3
   % mex StochasticPooling.cpp COMPFLAGS="/openmp $COMPFLAGS" CXXFLAGS="\$CFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp" -largeArrayDims 
end;

rand('state',0)
clear cnn;
%%
%网络设计
%type:选择网络类型。可选项�?
%		'i'为输入层;
%		'c'为卷积层;
%		's'为池化层;
%		'o'为输出层;
%outputmaps：该卷积层输出feature map 张数，在卷积层使用�?
%kernelsize：该卷积层卷积窗大小，在卷积层使用�?
%activation：卷积层�?��函数选择，在卷积层使用�?可�?项：
%		'sigm';
%		'tanh';
%		'ReLU';
%		'softplus';
%scale：池化层池化窗大小，在池化层使用�?
%method：池化层池化方法，在池化层使用�?可�?项：
%		'm':maxpooling; 
%		'a':average pooling; 
%		's':stochastic pooling ;
%objective:分类层分类方法，在分类层使用。可选项�?
%		'sigm';
%		'softmax';
%训练手写数字样本的简单网�?
cnn.layers = {
    struct('type', 'i') %input layer
    struct('type', 'c', 'outputmaps', 6, 'kernelsize', 5, 'activation', 'sigm') %convolution layer 
    struct('type', 's', 'scale', 2, 'method', 'a') %sub sampling layer 'm':maxpooling; 'a':average pooling; 's':stochastic pooling
    struct('type', 'c', 'outputmaps', 12, 'kernelsize', 5, 'activation', 'sigm') %convolution layer
    struct('type', 's', 'scale', 2, 'method', 'a') %subsampling layer
    struct('type','o','objective','softmax');
};
%%
%全连接层设计
%numsoutput：该层全连接节点�?
%activation：该层全连接可�?�?��函数类型。可选项�?
%		'sigm';
%		'tanh';
%		'ReLU';
%		'softplus';
%dropoutFraction：dropout方法取消节点概率（�?常为0.5，当�?时dropout无效�?
%dropconnectFraction：dropconnect方法取消节点概率（�?常为0.5，当�?时dropconnect无效�?
cnn.fc={
    %struct('type','f','numsoutput',50, 'activation', 'ReLU','dropoutFraction',0.5,'dropconnectFraction',0)
    %struct('type','f','numsoutput',50, 'activation', 'ReLU','dropoutFraction',0,'dropconnectFraction',0)
    %struct('type','f','numsoutput',100)
    %struct('type','f','numsoutput',30)
};

opts.alpha = 1;%更新步长
%opts.alphascale = 0.5; 
opts.batchsize = 50;%更新�?��参数�?��的样本批量大�?
opts.numepochs = 1;%迭代次数

opts.momSwitch=false;%momentum（动量）寻优方法�?��
if opts.momSwitch
    opts.mominit=0.5;%动量参数初始�?
    opts.momentum = 0.9;%动量增加�?
    opts.momIncrease = 1800;%多少次更新后动量增加
    cnn.iter = 1;%初�?计数�?
end
cnn.testing = false;%�?��训练网络为false

cnn = cnnsetup(cnn, train_x, train_y);%网络配置，参数初始化
cnn = cnntrain(cnn, train_x, train_y, opts);%训练网络

[train_er, train_bad] = cnntest(cnn, train_x, train_y);
[test_er, test_bad] = cnntest(cnn, test_x, test_y);%测试网络


%plot mean squared error
figure; plot(cnn.rL);%绘制LOSS函数�?
disp(['error rate of train set is ' num2str(train_er*100) '%']);
disp(['error rate of test set is ' num2str(test_er*100) '%']);

assert(test_er<0.12, 'Too big error');