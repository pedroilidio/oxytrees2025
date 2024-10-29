
function y2 = predict_svm_kron(y,ka,kb)
	% Support Vector Machine for link prediction.
	%
	% Usage:  predict_svm_kron(y,ka,kb)
	% where y = 0/1 link matrix
	%       ka = kernel for first dimension
	%       kb = kernel for second dimension
	%
	% This function can be used in cross_validate(@predict_svm_kron,...)
	
	addpath ./svm_light;
	c = 0.1;
	y2 = svm_learn_octave(y*2-1,ka,kb,c);
end
