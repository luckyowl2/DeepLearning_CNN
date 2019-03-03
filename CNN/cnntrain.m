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
    
    for i = 1 : opts.numepochs%��������
        
        kk = randperm(m);
        for l = 1 : numbatches%һ�ε����и��²�������
            
            disp(['epoch ' num2str(i) '/' num2str(opts.numepochs) 'batch ' num2str(l) '/' num2str(numbatches)]);
            
            batch_x = x(:, :, kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));
            batch_y = y(:,    kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));
            
            
            net = cnnff(net, batch_x);%ǰ�򴫲�
            net = cnnbp(net, batch_y);%���򴫲��������
            net = cnnapplygrads(net, opts);%��������
            
            if isempty(net.rL)
                
                net.rL(1) = net.L;
            end
            %������ʷ����Ա㻭ͼ
            net.rL(end + 1) = 0.99 * net.rL(end) + 0.01 * net.L;
            
            
            
        end
        
    end
    toc;
end
