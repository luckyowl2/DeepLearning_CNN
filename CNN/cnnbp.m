function net = cnnbp(net, y)
    %BP反向传播代码
    n = numel(net.layers);
    %计算最后分类层LOSS
    if strcmp(net.layers{n}.objective, 'sigm')
        %   error
        net.e = net.o - y;
        %  loss function
        net.L = gather(1/2* sum(net.e(:) .^ 2) / size(net.e, 2));

        net.od = net.e .* (net.o .* (1 - net.o));   %  output delta
    elseif strcmp(net.layers{n}.objective, 'softmax')
        %   error
        %net.e = -1 * (y - net.o)/size(net.layers{1}.a{1},3);
        net.e = -1 * (y - net.o);
        %  loss function
        %net.L = gather(-1 * sum(sum(y.*log(net.o)))/ size(net.e, 2));
        net.L = gather(-1 * mean(sum(y.*log(net.o))));
        %net.L = -1 * mean(sum(y.*log(net.o)));
        
        %%  backprop deltas
        net.od = net.e;   %  output delta
    end;
    %net.fvd = (net.ffW' * net.od) * size(net.layers{1}.a{1},3);%            %  feature vector delta
    net.fvd = (net.ffW' * net.od);
    %net.ffvd=net.fvd;
    %计算全连接层反向传播灵敏度
    if ~isempty(net.fc)
        for m=numel(net.fc):-1:1
            if strcmp(net.fc{m}.activation, 'sigm')
                net.fc{m}.fvd=net.fvd.*(net.fc{m}.o.*(1-net.fc{m}.o));
            elseif strcmp(net.fc{m}.activation, 'tanh')
                net.fc{m}.fvd=net.fvd.*(1-((net.fc{m}.o).^2));
            elseif strcmp(net.fc{m}.activation, 'ReLU')
                net.fc{m}.fvd = net.fvd .* (net.fc{m}.o > 0);
            elseif strcmp(net.fc{m}.activation, 'softplus')
                net.fc{m}.fvd=net.fvd.*net.fc{m}.dsoftplus;
            end
            net.fc{m}.fvd=net.fc{m}.fvd.*net.fc{m}.dropoutMask;
            net.fvd=(net.fc{m}.ffW)'*net.fc{m}.fvd;
        end
    end
    %当连接全连接层的最后一层为卷积层的情况
    if strcmp(net.layers{n-1}.type, 'c')         %  only conv layers has sigm function
        if strcmp(net.layers{n-1}.activation, 'sigm')
            net.fvd = net.fvd .* (net.fv .* (1 - net.fv));
        elseif strcmp(net.layers{n-1}.activation, 'tanh')
            net.fvd = net.fvd .* (1-((net.fv).^2));% need to be exploited
        elseif strcmp(net.layers{n-1}.activation, 'ReLU')
            net.fvd = net.fvd .* (net.fv > 0);
        elseif strcmp(net.layers{n-1}.activation, 'softplus')
            net.dsoftplus = [];
            for j = 1 : numel(net.layers{n-1}.dsoft)
                sa = size(net.layers{n-1}.dsoft{j});
                net.dsoftplus = [net.dsoftplus; reshape(net.layers{n-1}.dsoft{j}, sa(1) * sa(2), sa(3))];
            end;
            net.fvd = net.fvd .* net.dsoftplus;
        end;
    end
    
    %  reshape feature vector deltas into output map style
    %将向量转换为矩阵
    sa = size(net.layers{n-1}.a{1});
    fvnum = sa(1) * sa(2);
    for j = 1 : numel(net.layers{n-1}.a)
        net.layers{n-1}.d{j} = reshape(net.fvd(((j - 1) * fvnum + 1) : j * fvnum, :), sa(1), sa(2), sa(3));
    end
    %卷积和池化层的反向传播
    for l = (n - 2) : -1 : 1
        if strcmp(net.layers{l}.type, 'c')
            for j = 1 : numel(net.layers{l}.a)
                if strcmp(net.layers{l}.activation, 'sigm')
                    da = net.layers{l}.a{j} .* (1 - net.layers{l}.a{j});
                elseif strcmp(net.layers{l}.activation, 'tanh')
                    da = 1-((net.layers{l}.a{j}).^2);% need to be exploited
                elseif strcmp(net.layers{l}.activation, 'ReLU')
                    da = ( net.layers{l}.a{j} > 0);
                elseif strcmp(net.layers{l}.activation, 'softplus')
                    da = net.layers{l}.dsoft{j};
                end;
                if strcmp(net.layers{l + 1}.method, 'a')
                    net.layers{l}.d{j} = da .* (expand(net.layers{l + 1}.d{j} , [net.layers{l + 1}.scale net.layers{l + 1}.scale 1]) / net.layers{l + 1}.scale ^ 2);
                elseif strcmp(net.layers{l + 1}.method, 'm')
                    net.layers{l}.d{j} = da .* (expand(net.layers{l + 1}.d{j} , [net.layers{l + 1}.scale net.layers{l + 1}.scale 1]) .* net.layers{l + 1}.PosMatrix{j});
                elseif strcmp(net.layers{l + 1}.method, 's')
                    net.layers{l}.d{j} = da .* (expand(net.layers{l + 1}.d{j} , [net.layers{l + 1}.scale net.layers{l + 1}.scale 1]) .* net.layers{l + 1}.PosMatrix{j});
                end;
            end
        elseif strcmp(net.layers{l}.type, 's')
            for i = 1 : numel(net.layers{l}.a)
                z = zeros(size(net.layers{l}.a{1}));
                for j = 1 : numel(net.layers{l + 1}.a)
                     z = z + convn(net.layers{l + 1}.d{j}, rot180(net.layers{l + 1}.k{i}{j}), 'full');
                end
                net.layers{l}.d{i} = z;
            end
        end
    end

    %%  calc gradients
    for l = 2 : n - 1
        if strcmp(net.layers{l}.type, 'c')
            for j = 1 : numel(net.layers{l}.a)
                for i = 1 : numel(net.layers{l - 1}.a)
                    net.layers{l}.dk{i}{j} = convn(flipall(net.layers{l - 1}.a{i}), net.layers{l}.d{j}, 'valid') / size(net.layers{l}.d{j}, 3);
                end
                net.layers{l}.db{j} = sum(net.layers{l}.d{j}(:)) / size(net.layers{l}.d{j}, 3);
            end
        end
    end
    %最后分类层参数梯度计算
    if ~isempty(net.fc)
        net.dffW = net.od * (net.fc{numel(net.fc)}.o)' / size(net.od, 2);
        net.dffb = mean(net.od, 2);
    else
        net.dffW = net.od * (net.fv)' / size(net.od, 2);
        net.dffb = mean(net.od, 2);
    end
    
    %全连接层的参数梯度计算
    if ~isempty(net.fc)
        for m=numel(net.fc):-1:1
            %注意net.fc{m}.i=net.fc{m-1}.o
            net.fc{m}.dffW=net.fc{m}.fvd*(net.fc{m}.i)' / size(net.fc{m}.fvd, 2);
            net.fc{m}.dffb=mean(net.fc{m}.fvd,2);
        end
    end
    

    function X = rot180(X)
        X = flipdim(flipdim(X, 1), 2);
    end
end


