
function y2 = predict_nnn_pair_rls(y,ka,kb)
	% Nearest Neighbor for link prediction.
	%
	% Usage:  predict_nearest_profile(y,ka,kb)
	% where y = 0/1 link matrix
	%       ka = kernel for first dimension
	%       kb = kernel for second dimension
	%
	% This function can be used in cross_validate(@predict_nearest_profile,...)
	sv = @(x)sum(vec(x));
	
	[sa,idxa]=sort(ka,'descend');
	[sb,idxb]=sort(kb,'descend');
	[sc,idxc]=sort(y*y','descend');
	[sd,idxd]=sort(y'*y,'descend');
	knn = 15;
	w = 1;
	for j=1:knn
		wa = w*sa(j+1,:)';
		wb = w*sb(j+1,:);
		xs{j+0*knn} = bsxfun(@times, wa, y(idxa(j+1,:),:) );
		xs{j+1*knn} = bsxfun(@times, wb, y(:,idxb(j+1,:)) );
		%xs{j+2*knn} = w*y(idxc(j+1,:),:);
		%xs{j+3*knn} = w*y(:,idxd(j+1,:));
		w *= 1;
	end
	if 0
		for j=2:knn
			for i=0:1
				xs{j+i*knn} = ((j-1)*xs{j-1+i*knn}+xs{j+i*knn})/j;
			end
		end
	end
	n = length(xs);
	
	
	xx = zeros(n);   % xs'*xs
	xy = zeros(n,1); % xs'*y
	rankRLS = 0;
	if rankRLS
		ny=numel(y);
		%d = eye(ny)
		%p = ones(ny,1)/ny
		for i=1:n
			xy(i) = sv(xs{i} .* (y - rankRLS/ny^2*sv(y)));
			for j=1:n
				xx(i,j) = sv(xs{i} .* xs{j} - rankRLS/ny^2*sv(xs{i})*sv(xs{j}));
			end
		end
	else
		for i=1:n
			xy(i) = sv(xs{i} .* y);
			for j=1:n
				xx(i,j) = sv(xs{i} .* xs{j});
			end
		end
	end
	
	lambda = 50;
	a = inv(xx + lambda*eye(n)) * xy;
	y2 = 0;
	for i=1:n
		y2 = y2 + a(i)*xs{i};
	end
end
