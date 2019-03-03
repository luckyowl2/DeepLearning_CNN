function net = cnnff(net, x)
    %ǰ�򴫲�����
    n = numel(net.layers);
    net.layers{1}.a{1} = x;
    inputmaps = 1;
    global useGpu;
    %����ͳػ����ǰ�򴫲�
    for l = 2 : n-1   %  for each layer
        if strcmp(net.layers{l}.type, 'c')
            %  !!below can probably be handled by insane matrix operations
            for j = 1 : net.layers{l}.outputmaps   %  for each output map
                %  create temp output map
                if useGpu
                    z = gpuArray(single(zeros(size(net.layers{l - 1}.a{1}) - [net.layers{l}.kernelsize - 1 net.layers{l}.kernelsize - 1 0])));
                else
                    z = zeros(size(net.layers{l - 1}.a{1}) - [net.layers{l}.kernelsize - 1 net.layers{l}.kernelsize - 1 0]);
                end
                for i = 1 : inputmaps   %  for each input map
                    %  convolve with corresponding kernel and add to temp output map
                    z = z + convn(net.layers{l - 1}.a{i}, net.layers{l}.k{i}{j}, 'valid');
                end
                %  add bias, pass through nonlinearity
                if strcmp(net.layers{l}.activation, 'sigm')
                    net.layers{l}.a{j} = sigm(z + net.layers{l}.b{j});
                elseif strcmp(net.layers{l}.activation, 'tanh')
                    net.layers{l}.a{j} = tanh(z + net.layers{l}.b{j});% need to be exploited
                elseif strcmp(net.layers{l}.activation, 'ReLU')
                    net.layers{l}.a{j} = max(z + net.layers{l}.b{j},0);
                elseif strcmp(net.layers{l}.activation, 'softplus')
                    net.layers{l}.dsoft{j} = sigm(z + net.layers{l}.b{j});
                    net.layers{l}.a{j} = softplus(z + net.layers{l}.b{j});
                end
            end
            %  set number of input maps to this layers number of outputmaps
            inputmaps = net.layers{l}.outputmaps;
        elseif strcmp(net.layers{l}.type, 's')
            %{
            if net.testing
                for j = 1 : inputmaps
                    net.layers{l}.a{j} = StochaticTest(net.layers{l}.scale, net.layers{l - 1}.a{j});
                end;
            %}
                %  downsample
            if strcmp(net.layers{l}.method, 'a')%avarage pooing����
                for j = 1 : inputmaps
                    z = convn(net.layers{l - 1}.a{j}, ones(net.layers{l}.scale) / (net.layers{l}.scale ^ 2), 'valid');   %  !! replace with variable
                    net.layers{l}.a{j} = z(1 : net.layers{l}.scale : end, 1 : net.layers{l}.scale : end, :);
                    %net.layers{l}.dropoutMask{j} = (rand(size(net.layers{l}.a{j}))>net.dropoutFraction);
                    %net.layers{l}.a{j} = net.layers{l}.a{j} .* net.layers{l}.dropoutMask{j};
                end
            elseif strcmp(net.layers{l}.method, 'm')%max pooling����
                for j = 1 : inputmaps
                    if useGpu
                        [zz, maxPosition] = MaxPooling(gather(net.layers{l - 1}.a{j}),[net.layers{l}.scale net.layers{l}.scale]);%this is stupid. I don't know how to code CUDA parellel program so far.
                        net.layers{l}.a{j} = gpuArray(zz);
                    else
                        [net.layers{l}.a{j}, maxPosition] = MaxPooling(net.layers{l - 1}.a{j},[net.layers{l}.scale net.layers{l}.scale]);
                    end
                    maxPosition = sparse(ones(length(maxPosition),1),maxPosition,ones(length(maxPosition),1),1,numel(net.layers{l - 1}.a{j}));
                    net.layers{l}.PosMatrix{j} = reshape(full(maxPosition),size(net.layers{l - 1}.a{j})); % matlab��֧�ֶ�άϡ�����? -
                    %net.layers{l}.dropoutMask{j} = (rand(size(net.layers{l}.a{j}))>net.dropoutFraction);
                    %net.layers{l}.a{j} = net.layers{l}.a{j} .* net.layers{l}.dropoutMask{j};
                end
            elseif strcmp(net.layers{l}.method, 's')%Stochastic pooling����
                for j = 1 : inputmaps
                    if useGpu
                        [zz, StochasticPosition] = StochasticPooling(gather(net.layers{l - 1}.a{j}),[net.layers{l}.scale net.layers{l}.scale]);%this is stupid
                        net.layers{l}.a{j} = gpuArray(zz);
                    else
                        [net.layers{l}.a{j}, StochasticPosition] = StochasticPooling(net.layers{l - 1}.a{j},[net.layers{l}.scale net.layers{l}.scale]);
                    end
                    StochasticPosition = sparse(ones(length(StochasticPosition),1),StochasticPosition,ones(length(StochasticPosition),1),1,numel(net.layers{l - 1}.a{j}));
                    net.layers{l}.PosMatrix{j} = reshape(full(StochasticPosition),size(net.layers{l - 1}.a{j})); % matlab��֧�ֶ�άϡ�����? -
                    %net.layers{l}.dropoutMask{j} = (rand(size(net.layers{l}.a{j}))>net.dropoutFraction);
                    %net.layers{l}.a{j} = net.layers{l}.a{j} .* net.layers{l}.dropoutMask{j};
                end
            end
        end
    end

    %  concatenate all end layer feature maps into vector
    %����ȫ���Ӳ�ǰ��������������
    net.fv = [];
    for j = 1 : numel(net.layers{n-1}.a)
        sa = size(net.layers{n-1}.a{j});
        net.fv = [net.fv; reshape(net.layers{n-1}.a{j}, sa(1) * sa(2), sa(3))];
    end
    
    
    %ȫ���Ӳ��ǰ�򴫲�
    input=net.fv;
    if ~isempty(net.fc)
        net.fc{1}.i=net.fv;
        for m=1:numel(net.fc)
            if ~net.testing
                %dropConnect�����ļ���
                net.fc{m}.dropconnectMask = (rand(size(net.fc{m}.ffW))>net.fc{m}.dropconnectFraction);
                net.fc{m}.dcffW =net.fc{m}.ffW.*net.fc{m}.dropconnectMask;
            else
                net.fc{m}.dcffW =net.fc{m}.ffW;
            end
            if strcmp(net.fc{m}.activation, 'sigm')
                    net.fc{m}.o=sigm(net.fc{m}.dcffW * net.fc{m}.i + repmat(net.fc{m}.ffb, 1, size(net.fc{m}.i, 2)));
            elseif strcmp(net.fc{m}.activation, 'tanh')
                    net.fc{m}.o=tanh(net.fc{m}.dcffW * net.fc{m}.i + repmat(net.fc{m}.ffb, 1, size(net.fc{m}.i, 2)));
            elseif strcmp(net.fc{m}.activation, 'ReLU')
                    net.fc{m}.o=max(net.fc{m}.dcffW * net.fc{m}.i + repmat(net.fc{m}.ffb, 1, size(net.fc{m}.i, 2)),0);
            elseif strcmp(net.fc{m}.activation, 'softplus')
                    net.fc{m}.dsoftplus=sigm(net.fc{m}.dcffW * net.fc{m}.i + repmat(net.fc{m}.ffb, 1, size(net.fc{m}.i, 2)));%softplus����ĵ���?
                    net.fc{m}.o=softplus(net.fc{m}.dcffW * net.fc{m}.i + repmat(net.fc{m}.ffb, 1, size(net.fc{m}.i, 2)));
            end
            %dropOut�����ļ���
            if ~net.testing
                net.fc{m}.dropoutMask = (rand(size(net.fc{m}.o))>net.fc{m}.dropoutFraction);
                net.fc{m}.drop_o = net.fc{m}.o .* net.fc{m}.dropoutMask;
            else
                net.fc{m}.drop_o = net.fc{m}.o;
            end
            %input=net.fc{m}.o;
            if m==numel(net.fc)
                input=net.fc{m}.drop_o;
            else
                net.fc{m+1}.i=net.fc{m}.drop_o;
            end
        end
    end
    
    %  feedforward into output perceptrons
    %������ǰ�򴫲�
    if strcmp(net.layers{n}.objective, 'sigm')%sigm����
        net.o = sigm(net.ffW * input + repmat(net.ffb, 1, size(input, 2)));
    elseif strcmp(net.layers{n}.objective, 'softmax')%softmax����
        M = net.ffW*input;
        M = bsxfun(@plus, M, net.ffb);%���
        %�������δ����ͬ��������һ�δ���
        %M = net.ffW*input+ repmat(net.ffb, 1, size(input, 2));
        M = bsxfun(@minus, M, max(M, [], 1));%��ÿ�����ֵ
        M = exp(M);%��exp
        M = bsxfun(@rdivide, M, sum(M));%��һ��
        net.o = M;
    end

end
