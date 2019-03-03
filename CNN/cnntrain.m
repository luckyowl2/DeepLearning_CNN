function net = cnntrain(net, x, y, opts)
    global useGpu;
    m = size(x, 3);
    numbatches = m / opts.batchsize;
    if rem(numbatches, 1) ~= 0
        error('numbatches not integer');
    end
    if useGpu
        x=gpuArray(x);
        y=gpuArray(y);
    end
    net.rL = [];
    tic;
    
    for i = 1 : opts.numepochs%迭代次数
        
        kk = randperm(m);
        for l = 1 : numbatches%一次迭代中更新参数次数
            
            disp(['epoch ' num2str(i) '/' num2str(opts.numepochs) 'batch ' num2str(l) '/' num2str(numbatches)]);
            
            batch_x = x(:, :, kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));
            batch_y = y(:,    kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));
            
            
            net = cnnff(net, batch_x);%前向传播
            net = cnnbp(net, batch_y);%反向传播计算误差
            net = cnnapplygrads(net, opts);%参数更新
            
            if isempty(net.rL)
                
                net.rL(1) = net.L;
            end
            %保存历史误差以便画图
            net.rL(end + 1) = 0.99 * net.rL(end) + 0.01 * net.L;
            
            
            
        end
        
    end
    toc;
end
