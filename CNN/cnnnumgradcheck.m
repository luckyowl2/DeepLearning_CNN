function cnnnumgradcheck(net, x, y)
    epsilon = 1e-4;
    er      = 1e-8;
    n = numel(net.layers);
    if ~isempty(net.fc)
        m = numel(net.fc);
    end

    for j = 1 : numel(net.ffb)
        net_m = net; net_p = net;
        net_p.ffb(j) = net_m.ffb(j) + epsilon;
        net_m.ffb(j) = net_m.ffb(j) - epsilon;
        net_m = cnnff(net_m, x); 
        net_m = cnnbp(net_m, y);
        net_p = cnnff(net_p, x); 
        net_p = cnnbp(net_p, y);
        d = (net_p.L - net_m.L) / (2 * epsilon);
        e = abs(d - net.dffb(j));
        e
        if e > er
            error('numerical gradient checking failed');
        end
    end
    
    for i = 1 : size(net.ffW, 1)
        for u = 1 : size(net.ffW, 2)
            net_m = net; net_p = net;
            net_p.ffW(i, u) = net_m.ffW(i, u) + epsilon;
            net_m.ffW(i, u) = net_m.ffW(i, u) - epsilon;
            net_m = cnnff(net_m, x); net_m = cnnbp(net_m, y);
            net_p = cnnff(net_p, x); net_p = cnnbp(net_p, y);
            d = (net_p.L - net_m.L) / (2 * epsilon);
            e = abs(d - net.dffW(i, u));
            u
            e
            if e > er
                error('numerical gradient checking failed');
            end
        end
    end
    
    if ~isempty(net.fc)
        for l=m:-1:1
            if strcmp(net.fc{m}.type, 'f')
                
                for u = 1 : numel(net.fc{m}.ffb)
                    net_m = net; net_p = net;
                    net_p.fc{m}.ffb(u) = net_m.fc{m}.ffb(u) + epsilon;
                    net_m.fc{m}.ffb(u) = net_m.fc{m}.ffb(u) - epsilon;
                    net_m = cnnff(net_m, x); net_m = cnnbp(net_m, y);
                    net_p = cnnff(net_p, x); net_p = cnnbp(net_p, y);
                    d = (net_p.L - net_m.L) / (2 * epsilon);
                    e = abs(d - net.fc{m}.dffb(u));
                    u
                    e
                    if e > er
                        error('numerical gradient checking failed');
                    end
                end
                
                for i = 1 : size(net.fc{m}.ffW, 1)
                    for j = 1 : size(net.fc{m}.ffW, 2)
                        net_m = net; net_p = net;
                        net_p.fc{m}.ffW(i, j) = net_m.fc{m}.ffW(i, j) + epsilon;
                        net_m.fc{m}.ffW(i, j) = net_m.fc{m}.ffW(i, j) - epsilon;
                        net_m = cnnff(net_m, x); net_m = cnnbp(net_m, y);
                        net_p = cnnff(net_p, x); net_p = cnnbp(net_p, y);
                        d = (net_p.L - net_m.L) / (2 * epsilon);
                        e = abs(d - net.fc{m}.dffW(i, j));
                        j
                        e
                        if e > er
                            error('numerical gradient checking failed');
                        end
                    end
                end
            end
        end
    end

    for l = n-1 : -1 : 2
        if strcmp(net.layers{l}.type, 'c')
            for j = 1 : numel(net.layers{l}.a)
                net_m = net; net_p = net;
                net_p.layers{l}.b{j} = net_m.layers{l}.b{j} + epsilon;
                net_m.layers{l}.b{j} = net_m.layers{l}.b{j} - epsilon;
                net_m = cnnff(net_m, x); net_m = cnnbp(net_m, y);
                net_p = cnnff(net_p, x); net_p = cnnbp(net_p, y);
                d = (net_p.L - net_m.L) / (2 * epsilon);
                e = abs(d - net.layers{l}.db{j});
                j
                e
                if e > er
                    error('numerical gradient checking failed');
                end
                for i = 1 : numel(net.layers{l - 1}.a)
                    for u = 1 : size(net.layers{l}.k{i}{j}, 1)
                        for v = 1 : size(net.layers{l}.k{i}{j}, 2)
                            net_m = net; net_p = net;
                            net_p.layers{l}.k{i}{j}(u, v) = net_p.layers{l}.k{i}{j}(u, v) + epsilon;
                            net_m.layers{l}.k{i}{j}(u, v) = net_m.layers{l}.k{i}{j}(u, v) - epsilon;
                            net_m = cnnff(net_m, x); net_m = cnnbp(net_m, y);
                            net_p = cnnff(net_p, x); net_p = cnnbp(net_p, y);
                            d = (net_p.L - net_m.L) / (2 * epsilon);
                            e = abs(d - net.layers{l}.dk{i}{j}(u, v));
                            j
                            e
                            if e > er
                                error('numerical gradient checking failed');
                            end
                        end
                    end
                end
            end
        elseif strcmp(net.layers{l}.type, 's')
%            for j = 1 : numel(net.layers{l}.a)
%                net_m = net; net_p = net;
%                net_p.layers{l}.b{j} = net_m.layers{l}.b{j} + epsilon;
%                net_m.layers{l}.b{j} = net_m.layers{l}.b{j} - epsilon;
%                net_m = cnnff(net_m, x); net_m = cnnbp(net_m, y);
%                net_p = cnnff(net_p, x); net_p = cnnbp(net_p, y);
%                d = (net_p.L - net_m.L) / (2 * epsilon);
%                e = abs(d - net.layers{l}.db{j});
%                if e > er
%                    error('numerical gradient checking failed');
%                end
%            end
        end
    end
%    keyboard
end
