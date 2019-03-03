function [er, bad] = cnntest(net, x, y)
    %  feedforward
    net.testing =true;%开始测试
    net = cnnff(net, x);%前向传播
    [~, h] = max(net.o);
    [~, a] = max(y);
    bad = find(h ~= a);%比较准确率

    er = numel(bad) / size(y, 2);
end
