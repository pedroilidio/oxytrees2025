
function y2 = predict_mul_kernel(y,ka,kb)
	% Predict values with GP method
	%m = -mean(vec(y));
	m = -0.;
	%m=0;
	y -= m;
	if 1
		y2a = predict(y,ka);
		y2b = predict(y',kb)';
		y2 = (y2a+y2b)/2;
		%y2 = max(y2a,y2b);
		y2 = y2 + m;
	else
	
		a = 0.8;
		y2 = y;
		for it = 1:5
			y3 = a * y + (1-a) * y2;
			y2a = predict(y3,ka);
			y2b = predict(y3',kb)';
			y2 = (y2a+y2b)/2;
			%y2 = max(y2a,y2b);
		end
	end
end

function y2=predict(y,ka)
	%y2 = ka*inv(ka+eye(length(ka)))*y;
	%y2 = ka*y;
	%y2 = bsxfun(@times, ka*y, 1./sum(ka,2));
	
	%d = diag(sum(ka,2));
	ka2 = ka - diag(diag(ka));
	d = diag(sum(ka2,2));
	y2 = inv(d)*ka2*y;
	
	%s=0;t=1;
	%s=-1;t=0;
	%y2 = (ka+s*eye(length(ka)))*inv(ka+t*eye(length(ka)))*y;
	%y2 = (ka)*inv(ka-1*eye(length(ka)))*-y;
	return
	
	a = 0.5;
	y2 = y;
	for i=1:3
		y3 = a * y + (1-a) * y2;
		y2 = bsxfun(@times, ka*y3, 1./sum(ka,2));
	end
end
