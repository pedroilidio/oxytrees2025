% This script runs a variety of prediction functions on different datasets with different kernels

base_method = @predict_rls_kron;
alphas = [0,0.5,1];
datas = {'nr','gpcr','ic','e'};
loo = 1; % use leave one out or 10fold cv?

for i=1:numel(datas)
	data = datas{i};
	fprintf(stderr,'Dataset: %s\n',data); fflush(stderr);
	for a=alphas
		method = predict_with_graph_kernel(base_method,a);
		fprintf(stderr,'  method: %-60s',function_to_string(method)); fflush(stderr);
		stats = cross_validate(method,data, 'loo',loo, 'include_pr',1, 'include_roc',1);
		pr_curve = stats.pr_curve;
		roc_curve = stats.roc_curve;
		fprintf(stderr,'.\t(auc=%.4f)\n',stats.auc); fflush(stderr);
		save(sprintf('../results/pr-curve-%s-alpha-%.1f.mat',data,a), '-v4','pr_curve');
		save(sprintf('../results/roc-curve-%s-alpha-%.1f.mat',data,a), '-v4','roc_curve');
	end
end
