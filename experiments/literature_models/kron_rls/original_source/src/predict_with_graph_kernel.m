
function ff = predict_with_graph_kernel(fun, ba, bb)
	% Add a graph kernel to the given kernel before applying some link prediction method.
	%
	% Usage: predict_with_graph_kernel(@predict_rls_kron, b)()
	% where b is the weight of graph kernel, default b=0.5.
	% b=0 returns the original kernel, b=1 uses only the graph kernel.
	
	if nargin < 2, ba = 0.5; end
	if nargin < 3, bb = ba; end
	ff = @(varargin) predict_with_graph_kernel_(fun,ba,bb, varargin{:});
end

function y2 = predict_with_graph_kernel_(fun,ba,bb, y,ka,kb)
	% Graph based kernel
	ga = y*y';
	gb = y'*y;
	ga = ga / mean(diag(ga));
	gb = gb / mean(diag(gb));
	ga = exp(-kernel_to_distance(ga));
	gb = exp(-kernel_to_distance(gb));
	
	% Combine the kernels
	ka2 = (1-ba) * ka + ba * ga;
	kb2 = (1-bb) * kb + bb * gb;
	%ka2 = ka.^(1-ba) .* ga.^ba;
	%kb2 = kb.^(1-bb) .* gb.^bb;
	
	% Predict with the given function
	y2 = fun(y,ka2,kb2);
end

function d=kernel_to_distance(k)
	% Given a kernel matrix k=y*y', calculate a matrix of square Euclidean distances
	di = diag(k);
	d = repmat(di,1,length(k)) + repmat(di',length(k),1) - 2 * k;
end
