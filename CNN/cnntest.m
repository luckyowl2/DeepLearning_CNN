function [er, bad] = cnntest(net, x, y)
    %  feedforward
    net.testing =true;%��ʼ����
    net = cnnff(net, x);%ǰ�򴫲�
    [~, h] = max(net.o);
    [~, a] = max(y);
    bad = find(h ~= a);%�Ƚ�׼ȷ��

    er = numel(bad) / size(y, 2);
end
