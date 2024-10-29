
function y=generic_fun(depth,f,x,x2)
	% Apply a function to all elements in some object
	% aka. deep map.
	% Maximal depth can be specified
	if nargin == 3
		if depth > 0 && isstruct(x)
			y = structfun(@(xx)generic_fun(depth-1,f,xx),x,'UniformOutput',false);
		elseif depth > 0 && iscell(x)
			y = cellfun(@(xx)generic_fun(depth-1,f,xx),x,'UniformOutput',false);
		else
			y = f(x);
		end
	elseif nargin == 4
		% Zip
		if depth > 0 && isstruct(x)
			% TODO: handle missing fields
			y = x;
			fields = fieldnames(x2);
			for i = 1:numel(fields)
				field=fields{i};
				if isfield(x,field)
					y = setfield(y,field, generic_fun(depth-1,f,getfield(x,field),getfield(x2,field)));
				else
					error('structs have different members');
				end
			end
		elseif depth > 0 && iscell(x)
			% TODO: handle different sizes
			y = x;
			for i = 1:numel(x)
				y{i} = generic_fun(depth-1,f,x{i},x2{i});
			end
		else
			y = f(x,x2);
		end
	end
end
