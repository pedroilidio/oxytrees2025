
function y2 = predict_weighted_profile(y,ka,kb)
	% Weighted Profile method for link prediction.
	%
	% Usage:  predict_weighted_profile(y,ka,kb)
	% where y = 0/1 link matrix
	%       ka = kernel for first dimension
	%       kb = kernel for second dimension
	%
	% This function can be used in cross_validate(@predict_weighted_profile,...)
	
	y2 = (predict(y,ka) + predict(y',kb)') / 2;
end

function y2 = predict(y,k)
	if 1
		% exclude diagonal (?)
		k = k - diag(diag(k));
	end
	for i = 1:length(k)
		y2(i,:) = (k(i,:) * y) / sum(k(i,:));
	end
end
