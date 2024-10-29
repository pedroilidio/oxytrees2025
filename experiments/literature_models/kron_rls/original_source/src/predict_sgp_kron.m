
function y2 = predict_sgp_kron(y,ka,kb)
	addpath ./stochastic_gp;
	sigma = 1;
	[na,nb] = size(y);
	ka2 = ka + sigma*eye(na);
	kb2 = kb + sigma*eye(nb);
	ika = inv(ka2);
	ikb = inv(kb2);
	%ika -= diag(diag(ika));
	%ikb -= diag(diag(ikb));
	y2 = sgp_octave(y,ika,ikb);
end
