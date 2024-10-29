
function y2 = predict_svm(y,ka,kb)
	% Support Vector Machine for link prediction.
	%
	% Usage:  predict_svm(y,ka,kb)
	% where y = 0/1 link matrix
	%       ka = kernel for first dimension
	%       kb = kernel for second dimension
	%
	% This function can be used in cross_validate(@predict_svm,...)
	
	addpath ./svm_light;
	c = 0.5;
	% Predict rows
	for i=1:length(kb)
		y2(:,i) = svm_learn_octave(y(:,i)*2-1,ka,eye(1),c);
	end
	for i=1:length(ka)
		y3(i,:) = svm_learn_octave(y(i,:)'*2-1,kb,eye(1),c)';
	end
	y2 = (y2+y3) / 2;
end
