% This script runs a variety of prediction functions on different datasets with different kernels

function run_all_tests(loo)

methods ...
  = {%@predict_laprls, ...
     predict_with_graph_kernel(@predict_rls,0,0), ...
     predict_with_graph_kernel(@predict_rls,0.5,0.5), ...
     predict_with_graph_kernel(@predict_rls,1,1), ...
     predict_with_graph_kernel(@predict_rls_kron,0,0), ...
     predict_with_graph_kernel(@predict_rls_kron,0.5,0.5), ...
     predict_with_graph_kernel(@predict_rls_kron,1,1), ...
     predict_with_correlation_kernel(@predict_rls,1,1), ...
     predict_with_correlation_kernel(@predict_rls_kron,1,1) };
datas = {'nr','gpcr','ic','e'};
if nargin < 1
	loo = 1; % use leave one out or 10fold cv?
end

for i=1:numel(datas)
	data = datas{i};
	fprintf(stderr,'Dataset: %s\n',data); fflush(stderr);
	for k=1:numel(methods)
		method = methods{k};
		fprintf(stderr,'  method: %-60s',function_to_string(method)); fflush(stderr);
		stats = cross_validate(method,data, 'loo',loo);
		disp(stats);
		% Report progress
		auc  = stats.auc;  if isnumeric(auc),  auc  = sprintf('%.4f',auc);  end
		aupr = stats.aupr; if isnumeric(aupr), aupr = sprintf('%.4f',aupr); end
		fprintf(stderr,['.\t(auc=',auc,' aupr=',aupr,')\n']); fflush(stderr);
	end
end

end
