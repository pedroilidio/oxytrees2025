
function y2 = predict_rls_alt(y,ka,kb)
	sigma = 1;
	
	% Predict values with GP method
	[na,nb] = size(y);
	
	%kika = ka * inv(ka+sigma*eye(na));
	%kikb = inv(kb+sigma*eye(nb)) * kb;
	
	kika = ka / (ka+sigma*eye(na));
	kikb = (kb+sigma*eye(nb)) \ kb;
	
	y2b = y/2;
	
	for i=1:10
		y2a = kika * (y - y2b);
		y2b = (y - y2a) * kikb;
	end
	y2 = y2a+y2b;
end
