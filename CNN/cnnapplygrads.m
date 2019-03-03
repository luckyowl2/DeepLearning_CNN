function net = cnnapplygrads(net, opts)
    %opts.momentum = 0.95;
    %opts.momIncrease = 20;
    %cnn.iter = 1;
    %%
    %SGD+momentum�������·���
    %opts.momSwitchΪfalseʱ�����·���ΪSGD
    mom = 0;
    if opts.momSwitch 
        mom = opts.mominit;
        net.iter = net.iter +1;
        if net.iter >= opts.momIncrease
            mom = opts.momentum;
        end;
    end
    %���¾����ػ������
    for l = 2 : numel(net.layers)
        if strcmp(net.layers{l}.type, 'c')
            for j = 1 : numel(net.layers{l}.a)
                for ii = 1 : numel(net.layers{l - 1}.a)
                    net.layers{l}.vk{ii}{j} = mom * net.layers{l}.vk{ii}{j} + opts.alpha * net.layers{l}.dk{ii}{j};
                    net.layers{l}.k{ii}{j} = net.layers{l}.k{ii}{j} - net.layers{l}.vk{ii}{j};
                end
                net.layers{l}.vb{j} = mom * net.layers{l}.vb{j} + opts.alpha * net.layers{l}.db{j};
                net.layers{l}.b{j} = net.layers{l}.b{j} - net.layers{l}.vb{j};
            end
        end
    end
    %����ȫ���Ӳ����
    if ~isempty(net.fc)
        for m=1:numel(net.fc)
            net.fc{m}.vffW = mom * net.fc{m}.vffW + opts.alpha * (net.fc{m}.dffW.*net.fc{m}.dropconnectMask);
            net.fc{m}.ffW=net.fc{m}.ffW - net.fc{m}.vffW;
            net.fc{m}.vffb = mom * net.fc{m}.vffb + opts.alpha * net.fc{m}.dffb;
            net.fc{m}.ffb=net.fc{m}.ffb - net.fc{m}.vffb;
        end
    end
    %��������������
    net.vffW = mom * net.vffW + opts.alpha * net.dffW;
    net.ffW = net.ffW - net.vffW;
    net.vffb = mom * net.vffb + opts.alpha * net.dffb;
    net.ffb = net.ffb - net.vffb;
end

