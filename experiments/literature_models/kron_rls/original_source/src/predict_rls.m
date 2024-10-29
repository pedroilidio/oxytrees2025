
function y2 = predict_rls(y,ka,kb)
	% Regularized Least Squares for link prediction.
	%
	% Usage:  predict_rls(y,ka,kb)
	% where y = 0/1 link matrix
	%       ka = kernel for first dimension
	%       kb = kernel for second dimension
	%
	% This function can be used in cross_validate(@predict_rls,...)
	
	sigma = 1;
	
	% Subtract the mean value
	if 0
		mean_value = full(mean(vec(y)));
	else
		mean_value = 0;
	end
	y = y - mean_value;
	
	% Predict values with GP method
	[na,nb] = size(y);
	y2a = ka * ((ka+sigma*eye(na)) \ y);
	y2b = (y / (kb+sigma*eye(nb))) * kb;
	y2 = (y2a+y2b)/2 + mean_value;
end
