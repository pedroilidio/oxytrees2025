
function y2 = predict_lfl_sqp(y,ka,kb)
	[na,nb] = size(y);

	% initial weight vector
	k = 6;
	lambda = 0.2;
	w0 = 1e-1 * rand(k,na+nb);
	w0 = vec(w0);
	
	% My checkgrad
	if 0
		eps = 1e-7;
		dl = dloss(w0,k,na,nb,y,lambda);
		l = loss(w0,k,na,nb,y,lambda);
		for i=1:min(10,numel(w0))
			w2 = w0;
			w2(i) = w2(i) + eps;
			l2 = loss(w2,k,na,nb,y,lambda);
			dl2(i) = (l2-l) / eps;
			fprintf("%f = %f\n",dl(i),dl2(i));
		end
		fprintf("-----------------------\n");
	end
	w0 = sqp(w0, {@(w)loss(w,k,na,nb,y,lambda), @(w)dloss(w,k,na,nb,y,lambda)});
	y2 = eval(w0,k,na,nb);
end

function y = sigmoid(x)
	y = 1 ./ (1 + exp(-x));
end
function y = dsigmoid(x)
	y = exp(-x) ./ (1 + exp(-x)).^2;
end

function y2 = eval(w,k,na,nb)
	w = reshape(w,k,na+nb);
	wa = [ones(1,na); w(:,1:na)];
	wb = [w(:,na+1:end); ones(1,nb)];
	%_wa=wa(:,1:5)
	%_wb=wb(:,1:5)
	y2 = sigmoid(wa' * wb);
end

function l = loss(w,k,na,nb,y,lambda)
	w = reshape(w,k,na+nb);
	wa = [ones(1,na); w(:,1:na)];
	wb = [w(:,na+1:end); ones(1,nb)];
	y2 = sigmoid(wa' * wb);
	% 
	%l = sum(vec( -log(y.*y2 + (1-y).*(1-y2)) ));
	l = 0.5 * sum(vec( (y-y2).^2 ));
	l = l + 0.5 * lambda * sum(vec(w).^2);
end

function g = dloss(w,k,na,nb,y,lambda)
	w = reshape(w,k,na+nb);
	wa = [ones(1,na); w(:,1:na)];
	wb = [w(:,na+1:end); ones(1,nb)];
	y2 = sigmoid(wa' * wb);
	% 
	z = (y2 - y) .* dsigmoid(wa' * wb);
	ga = wb(2:end,:)   * z' + lambda * w(:,1:na);
	gb = wa(1:end-1,:) * z  + lambda * w(:,na+1:end);
	g = [ga gb];
	g = vec(g);
end

