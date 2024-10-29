
function y2 = predict_lfl(y,ka,kb)
	
	[na,nb] = size(y);
	shuffle = randperm(numel(y));
	tr = floor(numel(y)*0.9);
	
	u = vec(repmat((1:na)',1,nb))(shuffle);
	m = vec(repmat(1:nb,na,1))(shuffle);
	r = vec(y)(shuffle) + 1;
	%Tr.u = u(1:tr); Tr.m = m(1:tr); Tr.r = r(1:tr);
	%Te.u = u(tr+1:end); Te.m = m(tr+1:end); Te.r = r(tr+1:end);
	Tr.u = u; Tr.m = m; Tr.r = r;
	
	k = 40;
	eta = 1;
	lambda = 0.00;
	epochs = 500;
	loss = 'mse';
	
	%w = lflSGDOptimizer(Tr,Tr,k,eta,lambda,epochs,loss);
	w = lflSGDOptimizer(Tr,[],k,eta,lambda,epochs,loss);
	y2 = lflPredictor(w) - 1;
	return
	
	%[u m r]'
	
	_y=y(1:5,1:5)
	_y2=y2(1:5,1:5)
	%_wu=w.userW(:,:,1:5) % (latent,class,user)
	%_wm=w.movieW(:,:,1:5)
	%w
	_su=sum(y(1:5,:),2)
	_sm=sum(y(:,1:5),1)
	
	_e1=exp(squeeze(w.userW(:,1,:))' * squeeze(w.movieW(:,1,:)))
	_e2=exp(squeeze(w.userW(:,2,:))' * squeeze(w.movieW(:,2,:)))
end
