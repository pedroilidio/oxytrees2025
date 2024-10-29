
function y2=loo_rls_kron(y,ka,kb)
	[na,nb] = size(y);
	[va,la] = eig(ka);
	[vb,lb] = eig(kb);
	
	vva = va.^2;
	vvb = vb.^2;
	m1 = va' * y * vb;
	
	% Leave one out estimates
	if 0
		ss = 0.1:0.1:5;
		for i=1:length(ss)
			sigma2 = ss(i);
			
			%l = kron(diag(lb)',diag(la));
			%l = l ./ (l + sigma2);
			
			%l = kron(diag(lb)',diag(la));
			%l = l ./ kron(diag(lb)'+sigma2,diag(la)+sigma2);
			l = bsxfun(@(a,b)f(a,b,sigma2), diag(la), diag(lb)');
			
			% y2 = v * l * v' * y
			%m1 = va' * y * vb;
			m2 = m1 .* l;
			y2 = va * m2 * vb';
			
			% v.^2
			% dy2 = v .* v .* repmat(diag(l)) .* y
			dy2 = vva * l * vvb;
			%yloo = y2 - dy2;
			yloo = y2 - dy2.*y;
			
			err(i) = sum(vec((yloo - y).^2));
		end
	elseif 1
		%ss = 0.1:0.1:5;
		ss = [0.1:0.1:5, 5.5:0.5:50];
		for i=1:length(ss)
			sigma2 = ss(i);
			l = bsxfun(@(a,b)f(a,b,sigma2), diag(la), diag(lb)');
			%err(i) = loo_loss(y,m1,va,vb,vva,vvb,l);
			err(i) = loo_loss_auc(y,m1,va,vb,vva,vvb,l);
		end
		
		[~,best]=min(err);
		best_sigma=ss(best)
		l = bsxfun(@(a,b)f(a,b,ss(best)), diag(la), diag(lb)');
		y2 = va * (m1 .* l) * vb';
	end
	if 0
		ss2 = 0.1:0.1:3;
		for i=1:numel(ss2)
			sigma2 = ss2(i);
			ls{i} = bsxfun(@(a,b)f(a,b,sigma2), diag(la), diag(lb)');
		end
		
		w = ones(1,numel(ls)) ./ numel(ls);
		
		if 0
			[x,grad] = loo_loss_grad(w, y,m1,va,vb,vva,vvb, ls)
			for i=1:numel(ss)
				dw = 0.001;
				w2 = w;
				w2(i) = w2(i) + dw;
				x2 = loo_loss_grad(w2, y,m1,va,vb,vva,vvb, ls);
				cgrad(i) = (x2-x)/dw;
			end
			cgrad
		end
		
		[w,errs] = minimize(w, @loo_loss_grad, 10, y,m1,va,vb,vva,vvb, ls);
		
		% Make final prediction
		l = zeros(size(ls{1}));
		for i=1:numel(w)
			l = l + w(i) * ls{i};
		end
		m2 = m1 .* l;
		y2 = va * m2 * vb';
	end
	if 0
		w = 1;
		l = kron(diag(lb)',diag(la));
		
		if 0
			ss2 = 0.1:0.4:3;
			for i=1:numel(ss2)
				w = ss2(i);
				[x,gradi] = loo_loss_grad_kron(w, y,m1,va,vb,vva,vvb, l);
				dw = 0.00001;
				x2 = loo_loss_grad_kron(w+dw, y,m1,va,vb,vva,vvb, l);
				cgrad(i) = (x2-x)/dw;
				grad(i)=gradi;
			end
			grad
			cgrad
		end
		
		[w,errs] = minimize(w, @loo_loss_grad_kron, 10, y,m1,va,vb,vva,vvb,l);
		w
		
		% Make final prediction
		ll = l ./ (l + w);
		y2 = va * (m1 .* ll) * vb';
	end
	
	%hold on
	plot(ss,err);
	if 0
		plot(ss,err);
		hold on
		plot(ss,repmat(errs(end),size(ss)),'r')
		hold off
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
function err = loo_loss_auc(y,m1,va,vb,vva,vvb,l)
	yloo = loo_eval(y,m1,va,vb,vva,vvb,l);
	%err = sum(vec((yloo - y).^2));
	%err = -calculate_auc(vec(y),vec(yloo)); % This is wrong, it uses one auc curve over all LOO data
	err = -calculate_aupr(vec(y)',vec(yloo)'); % This is wrong, it uses one auc curve over all LOO data
end

function auc = calculate_auc(targets, predicts)
	% Calculate area under the ROC curve
	if nargin > 1
		[~,i] = sort(predicts,'descend');
		targets = targets(i);
	end
	auc = sum(cumsum(targets)(~targets));
	pos = sum(targets);
	neg = sum(~targets);
	auc = auc / (pos * neg);
end
function aupr = calculate_aupr(targets, predicts)
	% Calculate area under the Precission/recall curve
	if nargin > 1
		[~,i] = sort(predicts,'descend');
		targets = targets(i);
	end
	aupr = sum((cumsum(targets)./(1:numel(targets)))(~~targets));
	pos = sum(targets);
	aupr = aupr / pos;
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
