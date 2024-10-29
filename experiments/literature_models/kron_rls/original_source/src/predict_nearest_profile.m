
function y2 = predict_nearest_profile(y,ka,kb)
	% Nearest Neighbor for link prediction.
	%
	% Usage:  predict_nearest_profile(y,ka,kb)
	% where y = 0/1 link matrix
	%       ka = kernel for first dimension
	%       kb = kernel for second dimension
	%
	% This function can be used in cross_validate(@predict_nearest_profile,...)
	
	y2 = (predict(y,ka) + predict(y',kb)') / 2;
	%y2 = max(predict(y,ka), predict(y',kb)');
end

function y2 = predict(y,k)
	for i = 1:length(k)
		[~,near] = sort(k(i,:),'descend');
		for j=near
			if j==i, continue; end;
			y2(i,:) = y(j,:);
			break;
		end
	end
end
