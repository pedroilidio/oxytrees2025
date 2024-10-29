
function out=fold1(f,xs)
	% Fold a function over a cell array.
	% This is a left fold, i.e.
	%  fold1(f,{a,b,c}) = f(f(a,b),c)
	%
	% The cell array may not be empty
	
	if iscell(xs)
		out = xs{1};
		for i = 2:numel(xs)
			out = f(out,xs{i});
		end
	else
		error('fold1 expected a cell array');
	end
end
