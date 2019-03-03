function [x, mu, sigma] = zscore(x,dim)
    mu=mean(x,dim);	
    sigma=max(std(x,0,dim),eps);
	x=bsxfun(@minus,x,mu);
	x=bsxfun(@rdivide,x,sigma);
end
