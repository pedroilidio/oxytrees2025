
function ff = predict_with_graph_kernel4(fun, ba, bb)
	if nargin < 2, ba = 0.5; end
	if nargin < 3, bb = ba; end
	ff = @(varargin) predict_with_graph_kernel4_(fun,ba,bb, varargin{:});
end

function y2 = predict_with_graph_kernel4_(fun,ba,bb, y,ka,kb)
	% Graph based kernel
	ga = y*y';
	gb = y'*y;
	ga2 = ga / mean(diag(ga));
	gb2 = gb / mean(diag(gb));
	ga2 = exp(-kernel_to_distance(ga2));
	gb2 = exp(-kernel_to_distance(gb2));
	
	% 
	[va,la1] = eig(ka);
	[vb,lb1] = eig(kb);
	%
	%la2 = ba * diag(sum((va' * y).^2,2))  + (1-ba) * la1;
	%lb2 = bb * diag(sum((vb' * y').^2,2)) + (1-bb) * lb1;
	la2 = diag(diag(va'*ga*va));
	lb2 = diag(diag(vb'*gb*vb));
	la2 = la2 .* (length(la2) ./ trace(la2)); % normalize
	lb2 = lb2 .* (length(lb2) ./ trace(lb2));
	la2 = ba * la2 + (1-ba) * la1;
	%la2 = ba * diag(diag((va'*y)*(y'*va))) + (1-ba) * la1;
	lb2 = bb * lb2 + (1-bb) * lb1;
	
	% Combine the kernels
	ka2 = va * la2 * va';
	kb2 = vb * lb2 * vb';
	%ka2 = ka.^(1-ba) .* ga.^ba;
	%kb2 = kb.^(1-bb) .* gb.^bb;
	
	if 0
		%all1a = alignl(y,va,diag(la1))
		%all2a = alignl(y,va,diag(la2))
		
		al1a = align(ga,ka)
		al1a = align(ga2,ka)
		%al1b = align(gb,kb)
		al2a = align(ga,ka2)
		al2a = align(ga2,ka2)
		%al2b = align(gb,kb2)
	end
	
	% Predict with the given function
	y2 = fun(y,ka2,kb2);
end

function d=kernel_to_distance(k)
	% Given a kernel matrix k=y*y', calculate a matrix of square Euclidean distances
	di = diag(k);
	d = repmat(di,1,length(k)) + repmat(di',length(k),1) - 2 * k;
end

function x=alignl(y,v,l)
	vys = sum((v' * y).^2,2);
	yy = y * y';
	%fprintf("%f / sqrt(%f*%f) = ", l'*vys, sum(vec(yy.*yy)), sum(l.^2));
	x = l'*vys / sqrt(sum(vec(yy.*yy)) * sum(l.^2));
end

function x=align(a,b)
	a = vec(a);
	b = vec(b);
	%fprintf("%f / sqrt(%f*%f) = ", sum(a .* b), sum(a.*a), sum(b.*b));
	x = sum(a .* b) / sqrt(sum(a.*a)*sum(b.*b));
end
