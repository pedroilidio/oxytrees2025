% This script runs the RLSKron predictor with different weights for the graph kernel

datas = {'gpcr','ic','nr','e'};
step = 0.1;
as = 0:step:1;
bs = 0:step:1;
base_method = @predict_rls_kron;

for i=1:numel(datas)
	dataname = datas{i};
	data = load_dataset(dataname);
	fprintf(stderr,'Running weight tests for dataset %s\n',dataname); fflush(stderr);
	
	for i=1:numel(as)
		for j=1:numel(bs)
			method = predict_with_graph_kernel(base_method,as(i),bs(j));
			score{i,j} = cross_validate(method,data, 'stddevs',0);
			score{i,j}.a = as(i);
			score{i,j}.b = bs(j);
			fprintf(stderr,'.'); fflush(stderr);
		end
	end
	fprintf(stderr,'\n'); fflush(stderr);

	aucs  = cellfun(@(x)x.auc,score);
	auprs = cellfun(@(x)x.aupr,score);

	save(['../results/weight-tests-score-' dataname '.mat'],'score');
	save(['../results/weight-tests-aucs-' dataname '.mat'], '-v4','aucs');
	save(['../results/weight-tests-auprs-' dataname '.mat'],'-v4','auprs');
end

