
function y2 = predict_laprls(y,sa,sb)
	% The NetLapRLS method for link prediction, as described in
	%  ""
	% 
	% This function can be used in cross_validate(@predict_laprls,...)
	
	% Parameters as per the above paper
	ga1 = gb1 = 1;
	ga2 = gb2 = 0.01;
	ba  = bb  = 0.3;
	
	ypos = y > 0;
	
	ka = ypos*ypos';
	kb = ypos'*ypos;
	wa = (ga1*sa + ga2*ka) / (ga1+ga2);
	wb = (gb1*sb + gb2*kb) / (gb1+gb2);
	
	da = diag(sum(wa));
	db = diag(sum(wb));
	la = da^-0.5 * (da - wa) * da^-0.5;
	lb = db^-0.5 * (db - wb) * db^-0.5;
	
	fa = wa * inv(wa + ba*la*wa) * y;
	fb = wb * inv(wb + bb*lb*wb) * y';
	
	y2 = (fa+fb') / 2;
end
