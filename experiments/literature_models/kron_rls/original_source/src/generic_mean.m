
function mean=generic_mean(xs)
	% Mean of a cell array of structured values
	sum  = fold1(@(a,b)generic_fun(1,@plusNumbers,a,b),xs);
	mean = generic_fun(1,@(x)divNumbers(x,numel(xs)), sum);
end

function ans=plusNumbers(a,b)
	if ischar(a)
		if ~isequal(a,b)
			error(sprintf('mean of different strings: "%s", "%s"',a,b));
		end
		ans=a;
	else
		ans=a+b;
	end
end

function ans=divNumbers(a,b)
	if ischar(a)
		ans=a;
	else
		ans=a/b;
	end
end
