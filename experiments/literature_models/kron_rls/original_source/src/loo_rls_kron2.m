
function y2=loo_rls_kron2(y,ka,kb)
	[na,nb] = size(y);
	[va,la] = eig(ka);
	[vb,lb] = eig(kb);
	
	if 0
		
		for sigma=1:3
			l = bsxfun(@(a,b)(a.*b)./(a.*b+sigma), diag(la), diag(lb)');
			
			m1 = va' * y * vb;
			m2 = m1 .* l;
			y2s{sigma} = va * m2 * vb';
		end
		
		%y2s{1}(1:8,1:8)
		%y2s{2}(1:8,1:8)
		
		sigma = 2;
		sigmat = 1;
		%l = bsxfun(@(a,b)1./(a.*b+sigma), diag(la), diag(lb)');
		%m2 = m1 .* l;
		%w2 = va * m2 * vb';
		%
		w0 = va * (m1 .* bsxfun(@(a,b)1./(a.*b+1e-6), diag(la), diag(lb)')) * vb';
		w1 = va * (m1 .* bsxfun(@(a,b)1./(a.*b+1), diag(la), diag(lb)')) * vb';
		w19 = va * (m1 .* bsxfun(@(a,b)1./(a.*b+1.9), diag(la), diag(lb)')) * vb';
		w2 = va * (m1 .* bsxfun(@(a,b)1./(a.*b+2), diag(la), diag(lb)')) * vb';
		w21 = va * (m1 .* bsxfun(@(a,b)1./(a.*b+2.1), diag(la), diag(lb)')) * vb';
		w3 = va * (m1 .* bsxfun(@(a,b)1./(a.*b+3), diag(la), diag(lb)')) * vb';
		w4 = va * (m1 .* bsxfun(@(a,b)1./(a.*b+4), diag(la), diag(lb)')) * vb';
		w5 = va * (m1 .* bsxfun(@(a,b)1./(a.*b+5), diag(la), diag(lb)')) * vb';
		w6 = va * (m1 .* bsxfun(@(a,b)1./(a.*b+6), diag(la), diag(lb)')) * vb';
		w7 = va * (m1 .* bsxfun(@(a,b)1./(a.*b+7), diag(la), diag(lb)')) * vb';
		w10 = va * (m1 .* bsxfun(@(a,b)1./(a.*b+10), diag(la), diag(lb)')) * vb';
		%y2s21 = minimize(m3,@grad_rls_kron_it,10,y,ka,kb,sigmat);
		%grad_rls_kron_it(w2-1e-5,y,ka,kb,sigmat)
		fprintf("0.0 ");grad_rls_kron_it(w0,y,ka,kb,sigmat);
		fprintf("1.0 ");grad_rls_kron_it(w1,y,ka,kb,sigmat);
		fprintf("1.9 ");grad_rls_kron_it(w19,y,ka,kb,sigmat);
		fprintf("2.0 ");grad_rls_kron_it(w2,y,ka,kb,sigmat);
		fprintf("2.1 ");grad_rls_kron_it(w21,y,ka,kb,sigmat);
		fprintf("3.0 ");grad_rls_kron_it(w3,y,ka,kb,sigmat);
		fprintf("4.0 ");grad_rls_kron_it(w4,y,ka,kb,sigmat);
		fprintf("5.0 ");grad_rls_kron_it(w5,y,ka,kb,sigmat);
		fprintf("6.0 ");grad_rls_kron_it(w6,y,ka,kb,sigmat);
		fprintf("7.0 ");grad_rls_kron_it(w7,y,ka,kb,sigmat);
		fprintf("10  ");grad_rls_kron_it(w10,y,ka,kb,sigmat);
		fprintf("2-ε ");grad_rls_kron_it(w2-1e-5,y,ka,kb,sigmat);
		fprintf("2+ε ");grad_rls_kron_it(w2+1e-5,y,ka,kb,sigmat);
		
		y2s21 = minimize(w0,@grad_rls_kron_it,100,y,ka,kb,sigmat);
		%y2s21 = minimize_lbfgsb(w3,@grad_rls_kron_it,20,y,ka,kb,sigmat);
		%checkgrad(w2,@grad_rls_kron_it,1e-4,y,ka,kb,sigmat)
		% My checkgrad
		if 0
			eps = 1e-7;
			[l,dy] = grad_rls_kron_it(w2,y,ka,kb,sigmat);
			for i=1:min(10,numel(w2))
				ww = w2;
				ww(i) = ww(i) + eps;
				l2 = grad_rls_kron_it(ww,y,ka,kb,sigmat);
				dy2(i) = (l2-l) / eps;
				fprintf("%f = %f\n",dy(i),dy2(i));
			end
		end
	end
	if 0
		
		sigma = 0.1;
		
		ga = y*y';
		gb = y'*y;
		ga = ga / mean(diag(ga));
		gb = gb / mean(diag(gb));
		ga = exp(-kernel_to_distance(ga));
		gb = exp(-kernel_to_distance(gb));
		
		%kas{1} = eye(na);
		kas{1} = normalize_kernel2( ka );
		kas{2} = normalize_kernel2( exp(-0.0*kernel_to_distance(ga)) );
		kas{3} = normalize_kernel2( exp(-0.5*kernel_to_distance(ga)) );
		kas{4} = normalize_kernel2( exp(-0.8*kernel_to_distance(ga)) );
		kas{5} = normalize_kernel2( exp(-1.0*kernel_to_distance(ga)) );
		kas{6} = normalize_kernel2( exp(-1.5*kernel_to_distance(ga)) );
		kas{7} = normalize_kernel2( exp(-2.0*kernel_to_distance(ga)) );
		kas{8} = normalize_kernel2( exp(-4.0*kernel_to_distance(ga)) );
		kas{9} = normalize_kernel2( exp(-8.0*kernel_to_distance(ga)) );
		
		%kbs{1} = eye(nb);
		kbs{1} = normalize_kernel2( kb );
		%kbs{4} = normalize_kernel2( exp(-0.5*kernel_to_distance(gb)) );
		kbs{2} = normalize_kernel2( exp(-0*kernel_to_distance(gb)) );
		kbs{3} = normalize_kernel2( exp(-1.0*kernel_to_distance(gb)) );
		%kbs{5} = normalize_kernel2( exp(-2.0*kernel_to_distance(gb)) );
		
		clear c
		c.a = 1e-3 * ones(size(kas));
		c.b = 1e-3 * ones(size(kbs));
		c.a(4) = c.b(3) = 10;
		
		% checkgrad
		if 0
			[l, dy ] = grad_mkl(c,y,kas,kbs,sigma)
			[l2,dy2] = emperical_grad(c,@grad_mkl,y,kas,kbs,sigma)
		end
		if 0
			eps = 1e-7;
			[l,dy] = grad_mkl(c,y,kas,kbs,sigma);
			for i=1:min(10,numel(kas))
				cc = c;
				cc.a(i) = cc.a(i) + eps;
				l2 = grad_mkl(cc,y,kas,kbs,sigma);
				dy2.a(i) = (l2-l) / eps;
				fprintf("a%d:  %f = %f\n",i,dy.a(i),dy2.a(i));
			end
			for i=1:min(10,numel(kas))
				cc = c;
				cc.b(i) = cc.b(i) + eps;
				l2 = grad_mkl(cc,y,kas,kbs,sigma);
				dy2.b(i) = (l2-l) / eps;
				fprintf("b%d:  %f = %f\n",i,dy.b(i),dy2.b(i));
			end
		end
		
		cc = minimize(c,@grad_mkl,10,y,kas,kbs,sigma)
		cc2	 = minimize(c,@(cc)emperical_grad(cc,@grad_mkl,y,kas,kbs,sigma),10)
	end
	
	if 1
		
		ga = y*y';
		gb = y'*y;
		ga = ga / mean(diag(ga));
		gb = gb / mean(diag(gb));
		%ga = exp(-kernel_to_distance(ga));
		%gb = exp(-kernel_to_distance(gb));
		
		%kas{1} = eye(na);
		kas{1} = normalize_kernel2( ka );
		kas{2} = normalize_kernel2( eye(na) );
		kas{3} = normalize_kernel2( ga );
		%kas{4} = normalize_kernel2( exp(-0.5*kernel_to_distance(ga)) );
		kas{4} = normalize_kernel2( exp(-1.0*kernel_to_distance(ga)) );
		%kas{6} = normalize_kernel2( exp(-2.0*kernel_to_distance(ga)) );
		%kas{7} = normalize_kernel2( exp(-4.0*kernel_to_distance(ga)) );
		%kas{8} = normalize_kernel2( exp(-8.0*kernel_to_distance(ga)) );
		
		%kbs{1} = eye(nb);
		kbs{1} = normalize_kernel2( kb );
		kbs{2} = normalize_kernel2( eye(nb) );
		%kbs{3} = normalize_kernel2( gb );
		%kbs{4} = normalize_kernel2( exp(-0.5*kernel_to_distance(gb)) );
		kbs{3} = normalize_kernel2( exp(-1.0*kernel_to_distance(gb)) );
		%kbs{6} = normalize_kernel2( exp(-2.0*kernel_to_distance(gb)) );
		
		clear c
		c.sigma = 0;
		c.a = rand(size(kas)) * 1e-0;
		c.b = rand(size(kbs)) * 1e-0;
		%c.a = ones(size(kas)) / numel(kas);
		%c.b = ones(size(kbs)) / numel(kbs);
		%c.a(1) = c.b(1) = 1;
		
		% checkgrad
		if 0
			eps = 1e-7;
			[l,dy] = grad_mkl_loo(c,y,kas,kbs);
			for i=1:min(10,numel(kas))
				cc = c;
				cc.a(i) = cc.a(i) + eps;
				l2 = grad_mkl_loo(cc,y,kas,kbs);
				dy2.a(i) = (l2-l) / eps;
				fprintf("a%d:  %f = %f\n",i,dy.a(i),dy2.a(i));
			end
			for i=1:min(10,numel(kbs))
				cc = c;
				cc.b(i) = cc.b(i) + eps;
				l2 = grad_mkl_loo(cc,y,kas,kbs);
				dy2.b(i) = (l2-l) / eps;
				fprintf("b%d:  %f = %f\n",i,dy.b(i),dy2.b(i));
			end
		end
		if 0
			[l2,dy2] = emperical_grad(c,@grad_mkl_loo,y,kas,kbs)
		end
		
		%c.a = c.b = [-10 -10 10];
		%cc = c;
		%cc = minimize(c,@grad_mkl_loo,10,y,kas,kbs,sigma);
		cc = minimize(c,@(cc)emperical_grad(cc,@grad_mkl_loo,y,kas,kbs),10)
		%cc
		
		[~,y2] = grad_mkl_loo(cc,y,kas,kbs);
	end
	
	% Leave one out estimates
	if 0
		vva = va.^2;
		vvb = vb.^2;
		
		ss = 0.1:0.1:5;
		for i=1:length(ss)
			sigma2 = ss(i);
			l = bsxfun(@(a,b)f(a,b,sigma2), diag(la), diag(lb)');
			err(i) = loo_loss(y,m1,va,vb,vva,vvb,l);
		end
		
		[~,best]=min(err);
		%best_sigma=ss(best)
		l = bsxfun(@(a,b)f(a,b,ss(best)), diag(la), diag(lb)');
		y2 = va * (m1 .* l) * vb';
		
		
		hold on
		plot(ss,err);
	end	
end

function [loss,grad] = grad_rls_kron_it(w,y,ka,kb,sigma)
	% loss = ∑{i}(yi-ŷi) + sigma * w*K*w
	y2 = ka * w * kb; % K*vec(y)
	%y2(1:8,1:8)
	loss1 = sum(vec((y2-y).^2));
	loss2 = sum(vec( w .* (ka * w * kb) ));
	loss = loss1 + sigma * loss2;
	grad1 = 2 * ka * (y2 - y) * kb;
	grad2 = 2 * sigma * ka * w * kb;
	grad = grad1 + grad2;
	
	fprintf("  %.3f + %.3f * %.3f = %.3f\n",loss1,sigma,loss2,loss);
end

function [loss,egrad] = grad_mkl(c,y,kas,kbs,sigma)
	% Logistic weights
	eta.a = exp(c.a);
	eta.b = exp(c.b);
	sumeta.a = sum(eta.a);
	sumeta.b = sum(eta.b);
	neta.a = eta.a / sumeta.a;
	neta.b = eta.b / sumeta.b;
	%eta = c;
	
	% Build kernel combination
	ka = zeros(size(kas{1}));
	for i = 1:numel(kas)
		ka = ka + neta.a(i) * kas{i};
	end
	
	kb = zeros(size(kbs{1}));
	for i = 1:numel(kbs)
		kb = kb + neta.b(i) * kbs{i};
	end
	
	% Solve
	[va,la] = eig(ka);
	[vb,lb] = eig(kb);
	m1 = va' * y * vb;
	w = va * (m1 .* bsxfun(@(a,b)1./(a.*b+sigma), diag(la), diag(lb)')) * vb';
	y2 = ka * w * kb; % K*vec(y)
	
	loss1 = sum(vec((y2-y).^2));
	loss2 = sum(vec( w .* (ka * w * kb) ));
	loss = loss1 + sigma * loss2;
	
	for i = 1:numel(kas)
		grad1 = 2 * sum(vec( (kas{i} * w * kb) .* (y2 - y) ));
		grad2 = sigma * sum(vec( w .* (kas{i} * w * kb) ));
		grad.a(i) = grad1 + grad2;
	end
	for i = 1:numel(kbs)
		grad1 = 2 * sum(vec( (ka * w * kbs{i}) .* (y2 - y) ));
		grad2 = sigma * sum(vec( w .* (ka * w * kbs{i}) ));
		grad.b(i) = grad1 + grad2;
	end
	
	for i = 1:numel(kas)
		%egrad.a(i) = 0;
		%for j = 1:numel(kas)
		%	if i == j
		%		egrad.a(i) = egrad.a(i) + grad.a(j) * neta.a(i) * (1 - neta.a(j));
		%	else
		%		egrad.a(i) = egrad.a(i) + grad.a(j) * neta.a(i) * -neta.a(j);
		%	end
		%end
		egrad.a(i) = neta.a(i) * grad.a * ((1:numel(kas)==i) - neta.a)';
	end
	for i = 1:numel(kbs)
		%egrad.b(i) = grad.b(i);
		egrad.b(i) = neta.b(i) * grad.b * ((1:numel(kbs)==i) - neta.b)';
	end
	
	% make loo prediction
	vva = va.^2;
	vvb = vb.^2;
	l = bsxfun(@(a,b)(a.*b)./(a.*b+sigma), diag(la), diag(lb)');
	ll = l ./ (l + sigma);
	yloo = loo_eval(y,m1,va,vb,vva,vvb,ll);
	loo_err  = sum(vec((yloo - y).^2));
	
	fprintf("("); fprintf("%.1f ",neta.a); fprintf(")");
	fprintf("("); fprintf("%.1f ",neta.b); fprintf(") ");
	fprintf("%.3f + %.3f * %.3f = %.3f  loo %f\n",loss1,sigma,loss2,loss,loo_err);
end

function [err,y2] = grad_mkl_loo(c,y,kas,kbs)
	eta.a = exp(c.a);
	eta.b = exp(c.b);
	eta.a /= sum(eta.a);
	eta.b /= sum(eta.b);
	
	% build kernels
	ka = zeros(size(kas{1}));
	for i = 1:numel(kas)
		ka = ka + eta.a(i) * kas{i};
	end
	kb = zeros(size(kbs{1}));
	for i = 1:numel(kbs)
		kb = kb + eta.b(i) * kbs{i};
	end
	
	c.sigma = 1;
	%c.sigma = exp(c.sigma);
	
	% Solve
	[va,la] = eig(ka);
	[vb,lb] = eig(kb);
	vva = va.^2;
	vvb = vb.^2;
	m1 = va' * y * vb;
	l = bsxfun(@(a,b)(a.*b)./(a.*b+c.sigma), diag(la), diag(lb)');
	y2 = va * (m1 .* l) * vb';
	
	% make loo prediction
	%ll = l ./ (l + c.sigma);
	yloo = loo_eval(y,m1,va,vb,vva,vvb,l);
	err  = sum(vec((yloo - y).^2));
	
	
	% gradient of sigma
%	dll = -l ./ (l + c.sigma).^2;
%	dy = loo_eval(y,m1,va,vb,vva,vvb,dll);
%	grad.sigma = 2*sum(vec(dy .* (yloo - y)));
	
	% gradient of kernel a
	
	% kernel b
	for i = 1:numel(kas)
		grad.a(i) = 0;
	end
	for i = 1:numel(kbs)
		grad.b(i) = 0;
	end
	
	% Debug
	if 0
		fprintf("["); fprintf("%.1f ",c.sigma); fprintf("]");
		fprintf("("); fprintf("%.1f ",c.a); fprintf(")");
		fprintf("("); fprintf("%.1f ",c.b); fprintf(")");
		fprintf("  loo %f\n",err);
	end
end


function yloo = loo_eval(y,m1,va,vb,vva,vvb,l)
	m2 = m1 .* l;
	y2 = va * m2 * vb';
	dy2 = vva * l * vvb;
	yloo = y2 - dy2.*y;
end
function err = loo_loss(y,m1,va,vb,vva,vvb,l)
	yloo = loo_eval(y,m1,va,vb,vva,vvb,l);
	err = sum(vec((yloo - y).^2));
end

function [err,grad] = loo_loss_grad(w, y,m1,va,vb,vva,vvb,ls)
	% l = weighted sum of ls
	l = zeros(size(ls{1}));
	for i=1:numel(w)
		l = l + w(i) * ls{i};
	end
	yloo = loo_eval(y,m1,va,vb,vva,vvb,l);
	err  = sum(vec((yloo - y).^2));
	for i=1:numel(w)
		%grad(i) = sum(vec(ls{i}.*(yloo - y)));
		dy = loo_eval(y,m1,va,vb,vva,vvb,ls{i});
		grad(i) = 2*sum(vec(dy.*(yloo - y)));
	end
end

function [err,grad] = loo_loss_grad_kron(w, y,m1,va,vb,vva,vvb,l)
	ll = l ./ (l + w);
	yloo = loo_eval(y,m1,va,vb,vva,vvb,ll);
	err  = sum(vec((yloo - y).^2));
	dll = -l ./ (l + w).^2;
	dy = loo_eval(y,m1,va,vb,vva,vvb,dll);
	grad(1) = 2*sum(vec(dy .* (yloo - y)));
	
	% Make final prediction
	y2 = va * (m1 .* ll) * vb';
end

function [y,grad] = emperical_grad(x,fun,varargin)
	y = fun(x,varargin{:});
	ux = unwrap(x);
	for i = 1:length(ux)
		%eps = 1e-7;
		eps = max(1,abs(ux(i))) * 1e-6;
		ux2 = ux; ux2(i) = ux2(i) + eps;
		x2 = rewrap(x,ux2);
		y2 = fun(x2,varargin{:});
		grad(i) = (y2 - y) / eps;
	end
end

function r=f(a,b,sigma)
	%if a+b<1, r=-1; else r=-0.5; end
	%r = -1 ./ (a+b+sigma);
	%r = (a.*b) ./ (a+b+sigma);
	%r = a.^2.*b.^2;
	%r = -1 ./ sqrt(a.^2+b.^2+1);
	%r = -1 ./ (a.*b+1);
	%r = -1 ./ (sqrt(a.*b)+1);
	%r = -1 ./ (a.*b+a+b+1);
	%r = -1 ./ max((a+b),sigma);
	%r = 0.5 + (a.*b-sigma) ./ (4*sigma) - (a.*b-sigma).^2./(8*sigma^2);
	%r =  - (a.*b-sigma) ./ (4*sigma);
	r = a.*b ./ (a.*b+sigma); % ≈ RLS-Kron
	%r = a.*b ./ ((a+sigma).*(b+sigma));
	%r = a ./ ((a+sigma)) + b./(b+sigma); % ≈ RLS-Avg
	%r = (1 + a./(a+1)) .* (1 + b./(b+1));
	%r = (a./(a+1)) .* (b./(b+1));
	%r = -exp(-a.*b);
	%r = max(a./((a+1)) , b./(b+1) );
end

function d=kernel_to_distance(k)
	% Given a kernel matrix k=y*y', calculate a matrix of square Euclidean distances
	di = diag(k);
	d = repmat(di,1,length(k)) + repmat(di',length(k),1) - 2 * k;
end

function k2 = normalize_kernel2(k)
	k2 = k * (length(k) / trace(k));
end



function v = unwrap(s)
	v = [];   
	if isnumeric(s)
	  v = s(:);                        % numeric values are recast to column vector
	elseif isstruct(s)
	  v = unwrap(struct2cell(orderfields(s))); % alphabetize, conv to cell, recurse
	elseif iscell(s)
	  for i = 1:numel(s)             % cell array elements are handled sequentially
		v = [v; unwrap(s{i})];
	  end
	end                                                   % other types are ignored
end
function [s v] = rewrap(s, v)
	if isnumeric(s)
	  if numel(v) < numel(s)
		error('The vector for conversion contains too few elements')
	  end
	  s = reshape(v(1:numel(s)), size(s));            % numeric values are reshaped
	  v = v(numel(s)+1:end);                        % remaining arguments passed on
	elseif isstruct(s) 
	  [s p] = orderfields(s); p(p) = 1:numel(p);      % alphabetize, store ordering  
	  [t v] = rewrap(struct2cell(s), v);                 % convert to cell, recurse
	  s = orderfields(cell2struct(t,fieldnames(s),1),p);  % conv to struct, reorder
	elseif iscell(s)
	  for i = 1:numel(s)             % cell array elements are handled sequentially 
		[s{i} v] = rewrap(s{i}, v);
	  end
	end                                             % other types are not processed
end
