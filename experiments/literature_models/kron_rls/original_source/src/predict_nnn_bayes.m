
function y2 = predict_nnn_bayes(y,ka,kb)
	% Nearest Neighbor for link prediction.
	%
	% Usage:  predict_nearest_profile(y,ka,kb)
	% where y = 0/1 link matrix
	%       ka = kernel for first dimension
	%       kb = kernel for second dimension
	%
	% This function can be used in cross_validate(@predict_nearest_profile,...)
	
	y2 = (predict(y,ka) + predict(y',kb)') / 2;
	%y2 = max(predict(y,ka), predict(y',kb)');
end

function y2 = predict(y,k)
	[~,idx]=sort(k,'descend');
	if 1
		cc00=cc01=cc10=cc11=zeros(length(idx),1);
		sumvec=@(x)sum(vec(x));
	else
		cc00=cc01=cc10=cc11=zeros(length(idx),size(y,2));
		sumvec=@(x)sum(x);
	end
	for j=1:length(idx)
		cc00(j,:)=sumvec(~y(idx(j,:),:).*~y);
		cc01(j,:)=sumvec(~y(idx(j,:),:).* y);
		cc10(j,:)=sumvec( y(idx(j,:),:).*~y);
		cc11(j,:)=sumvec( y(idx(j,:),:).* y);
	end
	prb0=(cc01+1)./(cc00+cc01+2);
	prb1=(cc11+1)./(cc10+cc11+2);
	lprb0=log(prb0);
	lprb1=log(prb1);
	%lprb-=repmat(mean(lprb,2),1,2);
	%[lprb0 lprb1](1:15,:)'
	%prior=(sum(vec(y))+1)/numel(y)+2
	%lprior=log(prior)
	
	
	%lprb0=zeros(length(idx),1);lprb1=ones(length(idx),1);
	
	w=0.8;
	lprb0 .*= w.^(1:length(idx))';
	lprb1 .*= w.^(1:length(idx))';
	
	y2 = zeros(size(y));
	for i = 1:length(k)
		for jj=1:20
			j=idx(jj,i);
			if j==i, continue; end;
			%y2(i,:) = y2(i,:) + y(j,:) * lprb(j,2);
			y2(i,:) = y2(i,:) + y(j,:) .* lprb1(jj,:) + (1-y(j,:)) .* lprb0(jj,:);
			%y2(i,:) = y2(i,:) + lprb(jj,1) + y(j,:) * (lprb(jj,2) - lprb(jj,1));
			%y2(i,:) = y2(i,:) + y(j,:) * (lprb(jj,2) - lprb(jj,1));
		end
	end
end
