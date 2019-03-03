% function test_example_CNN
clear all;
close all;
clc;
%%
global useGpu;
useGpu = false;%GPU使用�?��，此处尚�?���?
%%
%人头样本
load head_data_128_v2_unit8.mat
nums_total_positive=19176;
nums_total_negative=62093;
nums_selected_positive=19000;
nums_selected_negative=19000;
nums_train=30000;
nums_predict=8000;
randPosi=randperm(nums_total_positive);
randNega=randperm(nums_total_negative);
selected_positive_data=positive_data(:,:,randPosi(1:nums_selected_positive));
selected_negative_data=negative_data(:,:,randNega(1:nums_selected_negative));
total_data(:,:,1:nums_selected_positive)=selected_positive_data;
total_data(:,:,nums_selected_positive+1:nums_selected_positive+nums_selected_negative)=selected_negative_data;
total_label=zeros(2,nums_selected_positive+nums_selected_negative);
total_label(1,1:nums_selected_positive)=1;
total_label(2,nums_selected_positive+1:nums_selected_positive+nums_selected_negative)=1;
randTotal=randperm(nums_selected_positive+nums_selected_negative);
if useGpu %使用GPU，样本类型为double�?
    train_x=single(total_data(:,:,randTotal(1:nums_train)));
    test_x=single(total_data(:,:,randTotal(nums_train+1:nums_train+nums_predict)));
    train_y=single(total_label(:,randTotal(1:nums_train)));
    test_y=single(total_label(:,randTotal(nums_train+1:nums_train+nums_predict)));
else
    train_x=total_data(:,:,randTotal(1:nums_train));
    test_x=total_data(:,:,randTotal(nums_train+1:nums_train+nums_predict));
    train_y=total_label(:,randTotal(1:nums_train));
    test_y=total_label(:,randTotal(nums_train+1:nums_train+nums_predict));
end
%}
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

%训练人头样本网络
cnn.layers = {
    struct('type', 'i') %input layer
    struct('type', 'c', 'outputmaps', 50, 'kernelsize', 9, 'activation', 'ReLU') %convolution layer
    struct('type', 's', 'scale', 2, 'method', 'm') %subsampling layer
    struct('type', 'c', 'outputmaps', 100, 'kernelsize', 7, 'activation', 'ReLU') %convolution layer
    struct('type', 's', 'scale', 2, 'method', 'm') %subsampling layer
    struct('type', 'c', 'outputmaps', 150, 'kernelsize', 4, 'activation', 'ReLU') %convolution layer
    struct('type', 's', 'scale', 2, 'method', 'm') %subsampling layer
    struct('type', 'c', 'outputmaps', 200, 'kernelsize', 3, 'activation', 'ReLU') %convolution layer
    struct('type', 's', 'scale', 2, 'method', 'm') %subsampling layer
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
    struct('type','f','numsoutput',100, 'activation', 'ReLU','dropoutFraction',0,'dropconnectFraction',0)
    %struct('type','f','numsoutput',50, 'activation', 'ReLU','dropoutFraction',0,'dropconnectFraction',0)%��������softmax,���һ��ȫ���Ӳ�����ReLU
    %struct('type','f','numsoutput',100)
    %struct('type','f','numsoutput',30)
};

opts.alpha = 0.00003;%ѧϰ��
%opts.alphascale = 0.5; 
opts.batchsize = 50;%��δ�С
opts.numepochs = 10;%������

opts.momSwitch=true;%momentum（动量）寻优方法�?��
if opts.momSwitch
    opts.mominit=0.5;%动量参数初始�?
    opts.momentum = 0.9;%动量增加�?
    opts.momIncrease = 1800;%多少次更新后动量增加
    cnn.iter = 1;%初�?计数�?
end
cnn.testing = false;%�?��训练网络为false

cnn = cnnsetup(cnn, train_x, train_y);%网络配置，参数初始化
cnn = cnntrain(cnn, train_x, train_y, opts);%训练网络


%��ֹout of memory
[train_er1, train_bad1] = cnntest(cnn, train_x(:,:,1:floor(nums_train/3)), train_y(:,1:floor(nums_train/3)));
[train_er2, train_bad2] = cnntest(cnn, train_x(:,:,(floor(nums_train/3)+1):(2*floor(nums_train/3))), train_y(:,(floor(nums_train/3)+1):(2*floor(nums_train/3))));
[train_er3, train_bad3] = cnntest(cnn, train_x(:,:,((2*floor(nums_train/3))+1):end), train_y(:,((2*floor(nums_train/3))+1):end));

[test_er, test_bad] = cnntest(cnn, test_x, test_y);%测试网络


%plot mean squared error
figure; plot(cnn.rL);%绘制LOSS函数�?
disp(['error rate of train set is ' num2str(((train_er1+train_er2+train_er3)/3)*100) '%']);
disp(['error rate of test set is ' num2str(test_er*100) '%']);

%assert(er<0.12, 'Too big error');