
function out=generic_mean_stddev(xs)
	% Calculate "mean (stddev)" of a cell array of structured values
	n = numel(xs);
	mean = generic_mean(xs);
	devs = generic_fun(1,@(xx)generic_fun(1,@(x,m)(x-m).^2,xx,mean), xs); % deviation from mean
	dev  = fold1(@(a,b)generic_fun(1,@plus,a,b),devs);
	if n > 1
		var = generic_fun(1,@(x)x/max(1,n-1), dev); % sample variance
	else
		var = generic_fun(1,@(x)inf, dev);
	end
	out = generic_fun(1,@format_mean_stddev, mean,var);
end

function out=format_mean_stddev(mean,var)
	stddev = sqrt(var);
	if length(mean) > 1
		out=mean;
	elseif stddev < 1e-4
		out=sprintf('%.4f (%.5f)',mean,stddev);
	else
		out=sprintf('%.4f (%.4f)',mean,stddev);
	end
end
