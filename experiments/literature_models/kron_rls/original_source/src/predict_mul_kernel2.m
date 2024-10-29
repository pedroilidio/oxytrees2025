
function y2 = predict_mul_kernel2(y,ka,kb)
	% Predict values with GP method
	y2a = predict(y,ka);
	y2b = predict(y',kb)';
	%y2 = (y2a+y2b)/2;
	%y2 = max(y2a,y2b);
	u = 1;
	y2 = (y2a.^u+y2b.^u).^(1/u);
end

function y2=predict(y,ka)
	la = 1;
	si = 1;
	i  = eye(size(ka));
	ik = inv(ka + la*i);
	%y2 = (ka + la*ik) \ ka * y;
	%y2 = (ka + la*ik) \ y;
	%y2 = (i + si*ik) \ y;
	%l = 1*diag(sum(ka)) - ka;
	%y2 = (i + si*l) \ y;
	
	%ka2 = ka + 0*i;
	%y2 = (ka2 + si*i) \ ka2 * y;
	
	%y2 = (i + si*ik) \ y;
	%y2 = y - si * ((ka + si*i) \ y);
	%mu = 0; y2 = mu*y + (ka + si*i) \ ((1-mu)*ka*y - mu*si*y);
	
	%mu1=mu2=0;
	mu1=0;mu2=0;z=-0.0;
	y2 = (ka + si*i) \ ((1-mu1)*ka*y - mu2*si*y + z);
	
	%j = diag(1*(sum(abs(ik))' - diag(abs(ik))) + 1);
	%jw = max(sum(abs(ik))' - diag(abs(ik))); j = jw * i;
	%y2 = (j + si*ik) \ y;
end
