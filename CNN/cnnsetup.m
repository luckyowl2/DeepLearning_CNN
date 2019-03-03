function net = cnnsetup(net, x, y)
%     assert(~isOctave() || compare_versions(OCTAVE_VERSION, '3.8.0', '>='), ['Octave 3.8.0 or greater is required for CNNs as there is a bug in convolution in previous versions. See http://savannah.gnu.org/bugs/?39314. Your version is ' myOctaveVersion]);
   %网络初始化
    inputmaps = 1;
    mapsize = size(squeeze(x(:, :, 1)));
    global useGpu;

    for l = 1 : numel(net.layers)   %  layer
        if strcmp(net.layers{l}.type, 's')%池化层的构建，参数初始化
            mapsize = mapsize / net.layers{l}.scale;
            assert(all(floor(mapsize)==mapsize), ['Layer ' num2str(l) ' size must be integer. Actual: ' num2str(mapsize)]);
            net.layers{l}.inputmaps = inputmaps;
            for j = 1 : inputmaps
                net.layers{l}.b{j} = 0;
            end
        end
        if strcmp(net.layers{l}.type, 'c')%卷积层的构建，参数初始化
            mapsize = mapsize - net.layers{l}.kernelsize + 1;
            fan_out = net.layers{l}.outputmaps * net.layers{l}.kernelsize ^ 2;
            net.layers{l}.inputmaps = inputmaps;
            for j = 1 : net.layers{l}.outputmaps  %  output map
                fan_in = inputmaps * net.layers{l}.kernelsize ^ 2;
                for i = 1 : inputmaps  %  input map
                    if useGpu
                        net.layers{l}.k{i}{j} = (gpuArray(single(rand(net.layers{l}.kernelsize))) - 0.5) * 2 * sqrt(6 / (fan_in + fan_out));
                        net.layers{l}.vk{i}{j} = gpuArray(single(zeros(net.layers{l}.kernelsize)));
                    else
                        net.layers{l}.k{i}{j} = (rand(net.layers{l}.kernelsize) - 0.5) * 2 * sqrt(6 / (fan_in + fan_out));
                        net.layers{l}.vk{i}{j} = zeros(net.layers{l}.kernelsize);
                    end;
                end
                if useGpu
                    net.layers{l}.b{j} = (gpuArray(single(rand(1,1))) - 0.5) * 2 * sqrt(6 / (fan_in + fan_out));
                    %net.layers{l}.b{j}=gpuArray.zeros(1,1);
                    net.layers{l}.vb{j} = gpuArray(single(zeros(1,1)));
                else
                    net.layers{l}.b{j} = (rand(1,1) - 0.5) * 2 * sqrt(6 / (fan_in + fan_out));
                    %net.layers{l}.b{j} = 0;
                    net.layers{l}.vb{j} = 0;
                end;
            end
            inputmaps = net.layers{l}.outputmaps;
        end
    end
    % 'onum' is the number of labels, that's why it is calculated using size(y, 1). If you have 20 labels so the output of the network will be 20 neurons.
    % 'fvnum' is the number of output neurons at the last layer, the layer just before the output layer.
    % 'ffb' is the biases of the output neurons.
    % 'ffW' is the weights between the last layer and the output neurons. Note that the last layer is fully connected to the output layer, that's why the size of the weights is (onum * fvnum)
    fvnum = prod(mapsize) * inputmaps;
    onum = size(y, 1);
    
    numsinput=fvnum;
    %全连接层的构建，参数初始化
    if ~isempty(net.fc)
     for m = 1 : numel(net.fc)
          if strcmp(net.fc{m}.type, 'f')      
              if useGpu
                  net.fc{m}.ffW=(gpuArray(single(rand(net.fc{m}.numsoutput, numsinput))) - 0.5) * 2 * sqrt(6 / (net.fc{m}.numsoutput + numsinput)); 
                  net.fc{m}.vffW = gpuArray(single(zeros(net.fc{m}.numsoutput, numsinput)));
                  net.fc{m}.ffb=gpuArray(single(zeros(net.fc{m}.numsoutput,1)));
                  net.fc{m}.vffb = gpuArray(single(zeros(net.fc{m}.numsoutput,1)));
              else
                  net.fc{m}.ffW=(rand(net.fc{m}.numsoutput, numsinput) - 0.5) * 2 * sqrt(6 / (net.fc{m}.numsoutput + numsinput));        
                  net.fc{m}.vffW = zeros(net.fc{m}.numsoutput, numsinput);
                  net.fc{m}.ffb=zeros(net.fc{m}.numsoutput,1);
                  net.fc{m}.vffb = zeros(net.fc{m}.numsoutput,1);
              end
          end
          numsinput=net.fc{m}.numsoutput;
     end
    end
    %分类层的构建，参数初始化
    if useGpu
        net.ffb = gpuArray(single(zeros(onum, 1)));
        net.vffb = gpuArray(single(zeros(onum, 1)));
        net.ffW = (gpuArray(single(rand(onum, numsinput))) - 0.5) * 2 * sqrt(6 / (onum + numsinput));
        net.vffW = gpuArray(single(zeros(onum, numsinput)));
    else
        net.ffb = zeros(onum, 1);
        net.vffb = zeros(onum, 1);
        net.ffW = (rand(onum, numsinput) - 0.5) * 2 * sqrt(6 / (onum + numsinput));
        net.vffW = zeros(onum, numsinput);
    end;
end
