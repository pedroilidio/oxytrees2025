
function y2 = predict_rls_cg(y,ka,kb)
	sigma = 1;
	%alpha = -1;
	alpha = 1;
	%mulK  = @(x) ka*x + x*kb;
	%mulK  = @(x) ka*x + x*kb + alpha*bsxfun(@plus, ka*mean(x,2), mean(x,1)*kb); % = [√ka,√kb] features, that obviously doesn't work very well.
	mulK  = @(x) ka*ka*x + x*kb*kb + alpha*bsxfun(@plus, ka*mean(ka*x,2), mean(x*kb,1)*kb); % = [ka,kb] features, that obviously doesn't work very well.
	mulKS = @(x) mulK(x) + sigma*x;
	y2 = conjgrad(mulKS, mulK(y), y);
end

function x = conjgrad(mulA,b,x)
	r = b-mulA(x);
	p = r;
	rsold = sum(vec(r.*r));
	
	for i=1:100
		Ap = mulA(p);
		alpha = rsold / sum(vec(p .* Ap));
		x = x + alpha*p;
		r = r - alpha*Ap;
		rsnew = sum(vec(r.*r));
		if sqrt(rsnew) < 1e-10
			  break;
		end
		p = r + rsnew/rsold * p;
		rsold = rsnew;
	end
end
