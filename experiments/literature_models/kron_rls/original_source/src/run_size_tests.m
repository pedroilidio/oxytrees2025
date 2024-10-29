
as = bs = [0.1 0.3 0.5 0.7 0.9];
%nas = 10:10:200;
%nbs = 10:10:210;
base_data = load_dataset('ic');
nas = nbs = [10 20 30 50 100 150 200];
[ma,mb] = size(base_data.y);
base_method = @predict_rls_kron;
num_repetitions = 5;

aucs = auprs = scores = cell();
for u=1:numel(nas)
	for v=1:numel(nbs)
		na = nas(u);
		nb = nbs(v);
		fprintf(stderr,'Running weight tests for dataset %d/%d\n',na,nb); fflush(stderr);
		
		score = cell();
		for i=1:numel(as)
			for j=1:numel(bs)
				method = predict_with_graph_kernel(base_method,as(i),bs(j));
				subscore = cell();
				for k = 1:num_repetitions
					data = base_data;
					ax = randperm(ma)(1:na);
					bx = randperm(mb)(1:nb);
					% Pick rows such that each row/column has a non-zero?
					ok = 0;
					while ~ok
						ok = 1;
						for aa=1:numel(ax)
							if ~any(base_data.y(ax(aa),bx))
								ax(aa) = randperm(ma)(1);
								ok = 0;
							end
						end
						for bb=1:numel(bx)
							if ~any(base_data.y(ax,bx(bb)))
								bx(bb) = randperm(mb)(1);
								ok = 0;
							end
						end
					end
					% Build dataset
					data.y = base_data.y(ax,bx);
					data.k1 = base_data.k1(ax,ax);
					data.k2 = base_data.k2(bx,bx);
					data.l1 = base_data.l1(ax);
					data.l2 = base_data.l2(bx);
					subscore{k} = cross_validate(method,data,'kernel', 'stddevs',0,'num_repetitions',1);
				end
				score{i,j} = generic_mean(subscore);
				score{i,j}.a = as(i);
				score{i,j}.b = bs(j);
				fprintf(stderr,'.'); fflush(stderr);
			end
		end
		fprintf(stderr,'\n'); fflush(stderr);
		aucs{u,v}  = cellfun(@(x)x.auc,score);
		auprs{u,v} = cellfun(@(x)x.aupr,score);
		scores{u,v} = score;
	end
end

save(['../results/size-tests-scores-ic-subsets.mat'],'scores');
save(['../results/size-tests-aucs-ic-subsets.mat'], 'aucs');
save(['../results/size-tests-auprs-ic-subsets.mat'],'auprs');

aucs_cat = zeros(numel(as)*numel(nas));
auprs_cat = zeros(numel(bs)*numel(nbs));
for u=1:numel(nas)
	for v=1:numel(nbs)
		xs = numel(as)*(u-1)+1 : numel(as)*(u);
		ys = numel(bs)*(v-1)+1 : numel(bs)*(v);
		aucs_cat(xs,ys) = aucs{u,v};
		auprs_cat(xs,ys) = auprs{u,v};
	end
end
save(['../results/size-tests-aucs-ic-concat.mat'], '-v4', 'aucs_cat');
save(['../results/size-tests-auprs-ic-concat.mat'], '-v4', 'auprs_cat');

% Plot
figure;
function im(x),imshow(x,[min(vec(x)),max(vec(x))]);end
for u=1:numel(aucs)
	subplot(size(aucs,1), size(aucs,2), u);
	im(aucs{u});
end
