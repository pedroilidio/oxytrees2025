
function interacts = find_new_interactions(dataset, n)
	% Return the top n new interactions for all datasets
	%
	% Example:
	%  find_new_interactions('nr',10)
	%  % this finds the 10 highest scoring new interactions with nuclear receptor proteins
	
	if nargin < 1, dataset = 'all'; end;
	if nargin < 2, n = 50; end;
	
	method = predict_with_graph_kernel(@predict_rls_kron);
	if isequal(dataset,'all')
		interacts = cat(2, ...
				new_interactions(method, 'e',    n), ...
				new_interactions(method, 'ic',   n), ...
				new_interactions(method, 'gpcr', n), ...
				new_interactions(method, 'nr',   n) ...
			);
		interacts = sort_interacts(interacts);
		interacts = interacts(1:n);
	else
		interacts = new_interactions(method, dataset, n);
	end
end

function interacts = new_interactions(method,data,n)
	% Return the top n new interactions for the given dataset
	% i.e. interactions that are 0 in data.y, but that have a high score
	% 
	% Result is a cellarray of structures
	
	if ~isstruct(data)
		data = load_dataset(data);
	end
	
	% Predicted scores
	y2 = method(data.y, data.k1, data.k2);
	y2(find(data.y)) = -Inf; % exclude existing interactions
	[score,pos] = sort(vec(y2),'descend');
	n = min(n,nnz(~data.y)); % maximum number of predictions
	high_scores = pos(1:n);
	
	% Matching labels
	interacts = cell(1,n);
	for i=1:n
		i1 = mod((pos(i)-1),size(y2,1))+1;
		i2 = floor((pos(i)-1)./size(y2,1))+1;
		interacts{i}.score   = y2(i1,i2);
		interacts{i}.target  = data.l1{i1};
		interacts{i}.drug    = data.l2{i2};
		interacts{i}.target_id = i1;
		interacts{i}.drug_id   = i2;
		interacts{i}.dataset = data.name;
		% Also find the nearest neighbors that use the same protein
		neighbors = find(data.y(i1,:));
		neighbor_dists = data.k2(i2,neighbors);
		[min_dist,min_who] = max(neighbor_dists);
		interacts{i}.near_drug_neighbor_id   = neighbors(min_who);
		interacts{i}.near_drug_neighbor_name = data.l2{neighbors(min_who)};
		interacts{i}.near_drug_neighbor_k    = min_dist;
		% Also find the nearest neighbors that use the same protein
		neighbors = find(data.y(:,i2));
		neighbor_dists = data.k1(i1,neighbors);
		[min_dist,min_who] = max(neighbor_dists);
		interacts{i}.near_target_neighbor_id   = neighbors(min_who);
		interacts{i}.near_target_neighbor_name = data.l1{neighbors(min_who)};
		interacts{i}.near_target_neighbor_k    = min_dist;
	end
end

function sorted = sort_interacts(interacts)
	% Sort a list of interactions by their .score
	sorted = sort_cell_by(@(x)x.score, interacts, 'descend');
end

function sorted = sort_cell_by(fun,xs,varargin)
	% Sort a cellarray xs by comparing fun(xs{i})
	scores = cellfun(fun,xs);
	[~,pos] = sort(scores,varargin{:});
	%sorted = arrayfun(@(i)xs{i},pos,'UniformOutput',false);
	sorted = xs(pos);
end
