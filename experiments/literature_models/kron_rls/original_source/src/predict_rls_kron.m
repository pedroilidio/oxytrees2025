
function y2 = predict_rls_kron(y,ka,kb)
	% Kronecker product Regularized Least Squares for link prediction.
	%
	% Usage:  predict_rls_kron(y,ka,kb)
	% where y = 0/1 link matrix
	%       ka = kernel for first dimension
	%       kb = kernel for second dimension
	%
	% This function can be used in cross_validate(@predict_rls_kron,...)
	
	sigma = 1;
	
	% Subtract the mean value?
	if 0
		mean_value = full(mean(vec(y)));
	else
		mean_value = 0;
	end
	y = y - mean_value;
	
	% Predict values with RLSKron method:
	% K  = ka⊗kb
	% y* = K / (K+σI) * y
	
	[va,la] = eig(ka);
	[vb,lb] = eig(kb);
	
	l = kron(diag(lb)',diag(la));
	l = l ./ (l + sigma);
	
	m1 = va' * y * vb;
	m2 = m1 .* l;
	y2 = va * m2 * vb';
	
	y2 = y2 + mean_value;
end
