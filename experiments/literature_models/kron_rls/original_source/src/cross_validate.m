
function stats = cross_validate(fun, data, varargin)
	% Run cross-validation test for link prediction.
	%
	% Usage:
	%   stats = cross_validate(fun, data [,options])
	% 
	% fun:  handle to the link prediction function to test
	% data: either a dataset structure as returned by load_dataset, or the name of a dataset.
	% mode: (optional) testing mode. Possible values:
	%         - 'kernel':   (default) Use the kernels from the dataset
	%         - 'nokernel': Use the identity matrix instead of the kernels of the dataset
	%         - 'ideal':    Use ideal kernels, i.e. y*y'
	% options: more options, in the form of 'key',value pairs
	%    'kernel',mode        use the given kernel
	%    'loo',1:             use leave one out cross-validation instead of 10-fold cv
	%    'num_folds',10:      number of folds for cv
	%    'num_repetitions',5: number of repetitions of the experiment
	%    'stddevs',1:         output 'mean(stddev)' statistics as opposed to mean only
	%    'negatives',1:       use +1/-1 with missing values as 0, as opposed to +1/0
	% 
	
	if nargin < 2
		help cross_validate;
		return;
	end
	if ~isstruct(data)
		data = load_dataset(data);
	end
	
	% options
	opts = struct(varargin{:});
	if ~isfield(opts,'kernel'), opts.kernel = 'given'; end
	
	% which kernels to use?
	if isequal(opts.kernel,'given') && isfield(data,'k1') && isfield(data,'k2')
		k1 = data.k1;
		k2 = data.k2;
	elseif isequal(opts.kernel,'eye')
		k1 = eye(size(data.y,1));
		k2 = eye(size(data.y,2));
	elseif isequal(opts.kernel,'kernel1') && isfield(data,'k1')
		k1 = data.k1;
		k2 = eye(size(data.y,2));
	elseif isequal(opts.kernel,'kernel2') && isfield(data,'k2')
		k1 = eye(size(data.y,1));
		k2 = data.k2;
	elseif isequal(opts.kernel,'ideal')
		k1 = ideal_kernel(data.y);
		k2 = ideal_kernel(data.y');
	elseif isequal(opts.kernel,'random')
		k1 = random_kernel(size(data.y,1));
		k2 = random_kernel(size(data.y,2));
	else
		error(['unknown kernel mode: ' opts.kernel]);
	end
	
	stats = do_cross_validate(fun, data.y, k1, k2, varargin{:});
	stats.data = data.name;
	stats.kernel = opts.kernel;
end

function k=ideal_kernel(y)
	% Ideal kernel is the outer product of the target y values we want to predict
	k=y*y';
	d=diag(diag(k).^-0.5);
	k=d*k*d + 0.001*eye(size(k));
end
function k=random_kernel(n)
	r=rand(n,n);
	k=r*r';
end

function stats = do_cross_validate(fun, y,k1,k2, varargin)
	% Set the random seed, so we get exactly the same folds every time
	rand('state',1234567890);

	% Parse arguments
	opts = struct(varargin{:});
	% leave one out?
	if ~isfield(opts,'loo'), opts.loo = 0; end
	% leave out entire rows, instead of doing CV per instance
	if ~isfield(opts,'rows'), opts.rows = 0; end
	if ~isfield(opts,'cols'), opts.cols = 0; end
	if ~isfield(opts,'rank_cols'), opts.rank_cols = opts.rows; end
	% negatives as -1 instead of 0
	if ~isfield(opts,'negatives'), opts.negatives = 0; end
	% Calculate statistics for a whole repetition at once, or for each fold separately?
	if ~isfield(opts,'merge'), opts.merge = ~opts.rank_cols; end
	if opts.loo
		% Leave one out
		if opts.rows
			opts.num_folds = size(y,1);
		elseif opts.cols
			opts.num_folds = size(y,2);
		else
			opts.num_folds = numel(y);
		end
		if ~isfield(opts,'num_repetitions'), opts.num_repetitions = 1; end
		% Calculate standard deviations of output?
		if ~isfield(opts,'stddevs'), opts.stddevs = 0; end
	else
		% 10-fold cross validation
		if ~isfield(opts,'num_folds'), opts.num_folds = 10; end
		if ~isfield(opts,'num_repetitions'), opts.num_repetitions = 5; end
		% Calculate standard deviations of output?
		if ~isfield(opts,'stddevs'), opts.stddevs = 1; end
	end
	
	% Split data into folds
	[na,nb] = size(y);
	if opts.rows
		division = repmat(1:opts.num_folds,1,ceil(na/opts.num_folds));
		for rep = 1:opts.num_repetitions
			rows = division(randperm(na));
			which_fold{rep} = repmat(rows,1,nb); % matlab is column major
		end
	elseif opts.cols
		division = repmat(1:opts.num_folds,1,ceil(nb/opts.num_folds));
		for rep = 1:opts.num_repetitions
			cols = division(randperm(nb));
			which_fold{rep} = vec(repmat(cols,na,1))';
		end
	else
		division = repmat(1:opts.num_folds,1,ceil(na*nb/opts.num_folds)); % = [1,2,3,1,2,3,1,2,3,...]
		for rep = 1:opts.num_repetitions
			which_fold{rep} = division(randperm(na*nb));
		end
	end
	% TODO: divide the positives and negatives separately?

	% Optimization: with leave one out we can test all negatives at once, since leaving them out does nothing
	% If you don't trust me, set this variable to 0 and observe that the result stays the same
	i_trust_this_reasoning = 1;
	if i_trust_this_reasoning && opts.loo && opts.merge && ~opts.rows && ~opts.cols && ~opts.negatives
		% Leave one out
		opts.num_folds = nnz(y)+1;
		which = zeros(size(y));
		which(find(y)) = 1:nnz(y);
		which(~y) = nnz(y)+1; % do the negatives in fold nnz(y)+1
		for rep = 1:opts.num_repetitions
			which_fold{rep} = which;
		end
	end
	
	% Run the tests
	for rep = 1:opts.num_repetitions
		if opts.merge
			prediction = zeros(na,nb);
		end
		
		time = 0;
		for fold = 1:opts.num_folds
			% Set values outside this fold to 0
			which = (which_fold{rep}==fold);
			y_fold = y;
			if opts.negatives
				y_fold = full(y_fold*2-1);
			end
			y_fold(which) = 0;
			% Predict
			tic;
			prediction_fold = fun(y_fold,k1,k2);
			time = time + toc;
			% Determine statistics: separately per fold
			if opts.merge
				prediction(which) = prediction_fold(which);
			elseif opts.rank_cols
				assert(opts.rows)
				for row=find(which(1:na))
					j = (rep-1)*na+row;
					all_stats{j} = calculate_stats(y(row,:), prediction_fold(row,:), varargin{:});
					all_stats{j}.runtime = time;
				end
				time = 0;
			else
				j = (rep-1)*opts.num_folds+fold;
				all_stats{j} = calculate_stats(y(which), prediction_fold(which), varargin{:});
				all_stats{j}.runtime = time;
				time = 0;
			end
		end
		
		% Determine statistics: merged
		if opts.merge
			all_stats{rep} = calculate_stats(vec(y)', vec(prediction)', varargin{:});
			all_stats{rep}.runtime = time;
		end
	end
	
	% Done
	if opts.stddevs
		stats = generic_mean_stddev(all_stats);
	else
		stats = generic_mean(all_stats);
	end
	if opts.loo
		stats.num_folds = 'loo';
	else
		stats.num_folds = opts.num_folds;
	end
	stats.num_repetitions = opts.num_repetitions;
	stats.fun = function_to_string(fun);
end
