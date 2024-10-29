
function stats=calculate_stats(targets, predicts, varargin)
	% Calculate statistics of predicitions.
	% Optionally, plot a ROC curve
	%
	% usage:
	%   plot_roc(y values ordered by prediction positives first)
	% or
	%   plot_roc(ys,predicts)
	%
	% ys        should be an array of {0,1}
	% predicts  should be an array of real values, same size as ys
	
	if nargin > 1 && numel(predicts)
		[~,i] = sort(predicts,'descend');
		targets = targets(i);
	end
	
	% Parse arguments
	opts = struct(varargin{:});
	if ~isfield(opts,'include_roc'), opts.include_roc = 0; end
	if ~isfield(opts,'include_pr'), opts.include_pr = 0; end
	if ~isfield(opts,'plot_roc'), opts.plot_roc = 0; end
	if ~isfield(opts,'plot_pr'), opts.plot_pr = 0; end
	if ~isfield(opts,'plot_style'), opts.plot_style = ''; end
	
	% ROC curve
	xs = [0 cumsum(~targets)]; % False positives
	ys = [0 cumsum(targets)];  % True positives
	maxx=max(xs); maxy=max(ys);
	xs = full(xs/maxx);
	ys = full(ys/maxy);
	
	if opts.plot_roc
		hold on;
		plot([0 1],[0 1],':r');
		plot(xs,ys,opts.plot_style);
		xlabel('false positives');
		ylabel('true positives');
		axis([0 1 0 1]);
		hold off;
	end
	if opts.include_roc
		stats.roc_curve = [xs;ys];
	end
	if opts.plot_pr
		xs = [0 cumsum(~targets)]; % False positives
		ys = [0 cumsum(targets)];  % True positives
		ps = ys./max(xs+ys,1); % precission
		rs = ys / max(ys); % recall
		hold on;
		plot(rs,ps,opts.plot_style);
		xlabel('Recall');
		ylabel('Precission');
		axis([0 1 0 1]);
		hold off;
	end
	if opts.include_pr
		xs = [0 cumsum(~targets)]; % False positives
		ys = [0 cumsum(targets)];  % True positives
		ps = ys./max(xs+ys,1); % precission
		rs = ys / max(ys); % recall
		stats.pr_curve = [ps;rs];
	end
	
	% Calculate area under the ROC curve and other statistics
	stats.auc  = calculate_auc(targets);
	stats.aupr = calculate_aupr(targets);
	if opts.include_roc
		stats.roc  = sample_curve(xs,ys,0:0.01:1);
	end
	
	% Statistics assuming that the first 1% is positive
	percentile = max(1,floor(numel(targets) * 0.01));
	tp = sum(targets(1:percentile));
	fp = sum(~targets(1:percentile));
	tn = sum(~targets(percentile+1:end));
	fn = sum(targets(percentile+1:end));
	stats.sensitivity_1_100 = tp/(tp+fn);
	stats.specificity_1_100 = tn/(tn+fp);
	stats.ppv_1_100         = tp/(tp+fp);
	stats.accuracy_1_100    = (tp+tn)/(tp+fp+tn+fn);
	
	% Statistics assuming that the number of positives is the target number of positives
	percentile = sum(targets);
	tp = sum(targets(1:percentile));
	fp = sum(~targets(1:percentile));
	tn = sum(~targets(percentile+1:end));
	fn = sum(targets(percentile+1:end));
	stats.fraction_pf    = percentile / numel(targets);
	stats.sensitivity_pf = tp/(tp+fn);
	stats.specificity_pf = tn/(tn+fp);
	stats.ppv_pf         = tp/(tp+fp);
	stats.accuracy_pf    = (tp+tn)/(tp+fp+tn+fn);
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
	cumsums = cumsum(targets)./(1:numel(targets));
	aupr = sum(cumsums(~~targets));
	pos = sum(targets);
	aupr = aupr / pos;
end

function sampleY = sample_curve(xs,ys, sampleX)
	sampleY = zeros(size(sampleX));
	j = 1;
	for i=1:numel(sampleX)
		x = sampleX(i);
		while xs(j+1) < x, j=j+1; end
		if xs(j+1)-xs(j) <= 0
			frac = 0.5;
		else
			frac = (x-xs(j)) / (xs(j+1)-xs(j));
		end
		sampleY(i) = (1-frac)*ys(j) + frac*ys(j+1);
	end
end
