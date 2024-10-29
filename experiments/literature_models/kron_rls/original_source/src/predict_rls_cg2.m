
function y2 = predict_rls_cg2(y,ka,kb)
	
	sigma = ones(size(y));
	%sigma = ones(size(y)) - 0.5 * y;
	%sigma = 2 * ones(size(y)) - 1.5 * y;
	
	if 0
		% GIP kernels
		[na,nb] = size(y);
		for j=1:nb
			w = ones(1,nb);w(j)=0;
			ka2{j} = 0.5*ka + 0.5*gip(y*diag(w)*y');
		end
		for i=1:na
			w = ones(1,na);w(i)=0;
			kb2{i} = 0.5*kb + 0.5*gip(y'*diag(w)*y);
		end
		
		mulK = @(x) doMulK(x,ka2,kb2);
		%mulK = @(x) doMulK_Kron(x,ka2,kb2);
	elseif 1
		%mulK  = @(x) ka*x + x*kb; % = RLS-avg
		%mulK  = @(x) 1./(1./(ka*x+1e-3) + 1./(x*kb+1e-3) + 1e-3);
		%jka=1./ka;jkb=1./kb;mulK  = @(x) 1./(jka*(1./x) + (1./x)*jkb);
		%mulK  = @(x) ka*x + x*kb + ka*x*kb;
		%mulK  = @(x) min(ka*x, x*kb);
		%mulK  = @(x) ka*x*kb; % = RLS-Kron
		mulK  = @(x) ka*x*kb; % = RLS-Kron
		%mulK  = @(x) (ka*x.^2*kb) .^ 0.5;
		%mulK  = @(x) ka*x + 2*x*kb;
	else
		% Ranking version: use KLK instead of K
		% W = vec(y)==vec(y)' = bsxfun(@(a,b)a==b,y,y')
		% L = D-W = diag(sum(W)) - W = D - PP' = y'*y+(1-y')*(1-y)
		numPos = nnz(y);
		numNeg = numel(y) - nnz(y);
		%sigma *= length(y); % attempt at scaling
		%sigma = 100;
		d = y * numPos + (1-y) * numNeg;
		d = d + sigma;
		
		mulK1 = @(x) ka*x*kb; % = RLS-Kron
		%mulL  = @(x) d.*x - y*sum(vec(y.*x)) - (1-y)*sum(vec((1-y).*x));
		%a = 1.1;sigma=1;
		%mulL = @(x) (1-a)*x - y*(a*sum(vec(y.*x))/numPos) - (1-y)*(a*sum(vec((1-y).*x))/numNeg);
		%mulL = @(x) x;
		%a=0.1; sigma=5;
		%a=0.; sigma=2;
		a=1.;sigma=0.05;
		mulL = @(x) 1*x - a*ones(size(x))*sum(vec(x))/numel(x);
		%mulK  = @(x) mulK1(mulL(mulK1(x)));
		mulK  = @(x) mulL(mulK1(x));
		%mulK  = @(x) mulK1(mulL(x));
		
		mulKS = @(x) mulK(x) + sigma.*x;
		%y2 = conjgrad(mulKS, mulK1(mulL(y)), y);
		%y2 = conjgrad(mulKS, mulL(mulK1(y)), y);
		y2 = mulK1(conjgrad(mulKS, mulL(y), zeros(size(y))));
		return
	end
	
	mulKS = @(x) mulK(x) + sigma.*x;
	y2 = conjgrad(mulKS, mulK(y), y);
end

function y = doMulK(x,ka2,kb2)
	[na,nb] = size(x);
	for j=1:nb
		y1(:,j) = ka2{j} * x(:,j);
	end
	for i=1:na
		y2(i,:) = x(i,:) * kb2{i};
	end
	y = y1+y2;
end

function y = doMulK_Kron(x,ka2,kb2)
	[na,nb] = size(x);
	y = x;
	for j=1:nb
		y(:,j) = ka2{j} * y(:,j);
	end
	for i=1:na
		y(i,:) = y(i,:) * kb2{i};
	end
end

function ga = gip(yy,gamma)
	if nargin<2, gamma=1; end;
	ga = yy / mean(diag(yy));
	ga = exp(-gamma*kernel_to_distance(ga));
end

function d=kernel_to_distance(k)
	% Given a kernel matrix k=y*y', calculate a matrix of square Euclidean distances
	di = diag(k);
	d = repmat(di,1,length(k)) + repmat(di',length(k),1) - 2 * k;
end


function x = conjgrad(mulA,b,x)
	r = b - mulA(x);
	p = r;
	rsold = sum(vec(r.*r));
	
	for i=1:30
		Ap = mulA(p);
		alpha = rsold / sum(vec(p .* Ap));
		x = x + alpha*p;
		r = r - alpha*Ap;
		rsnew = sum(vec(r.*r));
		if sqrt(rsnew) < 1e-10
			  break;
		end
		p = r + rsnew/rsold * p;
		rsold = rsnew;
	end
end
