
function y2 = predict_rls_cg_cv(y,ka,kb)
	% Simple conjugate gradient, with cross-validation for parameter fitting.
	
	%loss = @square_loss;
	loss = @rank_rls_loss;
	%loss = @calculate_aupr;
	
	%method = @predict_rls_cg;
	%method = @predict_rank_rls_cg;
	%method = @predict_rank_rls_cg2;
	method = @predict_rls_plus_cg;
	
	%sigma = pick_sigma_cv(method,loss, y,ka,kb);
	sigma = 1;
	
	y2 = method(y,ka,kb,sigma);
end

% ------------------------------------------------------------------------------

function best_sigma = pick_sigma_cv(train,test, y,ka,kb)
	% Cross validation to pick sigma
	% Possible sigmas
	%sigmas = 2 .^ (-5:5);
	sigmas = 2 .^ (-4:0.5:4);
	%sigmas = 2 .^ (-2:0.5:2);
	% Split into folds
	num_folds = 5;
	divisions = repmat(1:num_folds,1,ceil(numel(y)/num_folds));
	which_fold = divisions(randperm(numel(y)));
	% CV
	scores = zeros(size(sigmas));
	for i = 1:numel(sigmas)
		sigma = sigmas(i);
		%y4 = zeros(size(y));
		score = 0;
		for f = 1:num_folds
			which = (which_fold==f);
			y2 = y; y2(which) = 0;
			y3 = train(y2,ka,kb, sigma);
			%y4(which) = y3(which);
			score = score + test(y(which),y3(which));
		end
		scores(i) = score;
		if 0
			printf("%.2f %.4f\n", sigmas(i),scores(i));
		end
	end
	% find best sigma
	[~,i] = max(scores);
	best_sigma = sigmas(i)
end

function l = square_loss(y1,y2)
	l = -sum(sumsq(y1 - y2));
end

function l = rank_rls_loss(y1,y2)
	dy = y1 - y2;
	% sum_ij (dy{i} - dy{j})² = sum_ij dy(i)² + dy(j)² - dy(i)dy(j)
	%  = 2*n*sum_i(dy(i)²) + sum dy * sum dy
	l=0;for i =1:numel(y1), l = l + 0.5*sumsq(dy - dy(i)); end
	l = numel(y1)*sumsq(dy) - sum(dy).^2;
	l = -l;
end

function auc = calculate_auc(targets, predicts)
	% Calculate area under the ROC curve
	if nargin > 1
		[~,i] = sort(predicts,'descend');
		targets = targets(i);
	end
	
	%for i=1:n
	%	if targets(i)
	%		goods = goods + 1
	%	else
	%		auc = auc + goods;
	%	end
	%end
	cumsums = cumsum(targets);
	auc = sum(cumsums(~targets));
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
	
	%for i=1:n
	%	if targets(i)
	%		goods = goods + 1
	%       aupr = aupr + good/(good+bad);
	%	else
	%		bad = bad + 1;
	%	end
	%end
	total = reshape(1:numel(targets), size(targets));
	cumsums = (cumsum(targets)./total);
	aupr = sum(cumsums(~~targets));
	pos = sum(targets);
	aupr = aupr / pos;
end

% ------------------------------------------------------------------------------

function y2 = predict_rls_cg(y,ka,kb, sigma)
	% Simple conjugate gradient kernel method
	mulK  = @(x) ka*x*kb; % = RLS-Kron
	mulKS = @(x) mulK(x) + sigma.*x;
	y2 = conjgrad(mulKS, mulK(y), y);
end

function y2 = predict_rls_plus_cg(y,ka,kb, sigma)
	% Simple conjugate gradient kernel method
	[na,nb] = size(y);
	%mulK  = @(x)  ka*x*kb - 2*(ka*x*ones(nb)/nb + ones(na)*x*kb/na); % makes no sense
	%mulK  = @(x) ka*x + x*kb;
	mulK  = @(x)  ka*x + x*kb - 1*(ka*x*ones(nb)/nb + ones(na)*x*kb/na); % ranking
	%mulK  = @(x)  x + ka*x + x*kb + ka*x*kb + ka*ka*x + ka*ka*x*kb + x*kb*kb + ka*x*kb*kb + ka*ka*x*kb*kb; %sigma = 5;
	mulKS = @(x) mulK(x) + sigma.*x;
	y2 = conjgrad(mulKS, mulK(y), y);
end

function y2 = predict_rank_rls_cg(y,ka,kb, sigma)
	% Simple conjugate gradient kernel method
	a=1.;
	mulL   = @(x) 1*x - a*ones(size(x))*sum(vec(x))/numel(x);
	mulK   = @(x) ka*x*kb; % = RLS-Kron
	mulLKS = @(x) mulL(mulK(x)) + sigma.*x;
	y2 = mulK(conjgrad(mulLKS, mulL(y), zeros(size(y))));
end

function y2 = predict_rank_rls_cg2(y,ka,kb, sigma)
	% Simple conjugate gradient kernel method
	a=1.;
	mulL   = @(x) 1*x - a*ones(size(x))*sum(vec(x))/numel(x);
	mulK   = @(x) ka*x*kb; % = RLS-Kron
	mulLKS = @(x) mulL(mulK(x)) + sigma.*x;
	y2 = mulK(conjgrad(mulLKS, mulL(y), zeros(size(y))));
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
