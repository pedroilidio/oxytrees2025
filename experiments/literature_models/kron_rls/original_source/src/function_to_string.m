
function out = function_to_string(x)
	% Convert a value, in particular a function, to a string
	% Gives a better result for anonymous functions compared to func2str
	%
	% Usage: function_to_string(@some_function_handle)
	
	if isa(x,'function_handle')
		info = functions(x);
		out = info.function;
		
		if isfield(info,'workspace')
			% subsittude workspace
			%  replace  @() stuff
			%  by 
			ws = info.workspace;
			if iscell(ws), ws = ws{1}; end
			ns = fieldnames(ws);
			for i=1:numel(ns)
				name = ns{i};
				if name(1)=='.', continue; end
				value = getfield(ws, ns{i});
				value = function_to_string(value);
				out = regexprep(out,['\b',name,'\b'],value);
			end
			% Try to clean up the result a bit
			out = regexprep(out,', varargin {:}','');
			out = regexprep(out,' \(','(');
			out = regexprep(out,' \{','{');
			out = regexprep(out,'^@\(\)\s*(\S*)_\(','$1(');
		end
	elseif ischar(x)
		out = x;
	elseif iscell(x)
		out = '{';
		for i=1:numel(x)
			if i>1, out = [out,',']; end
			value = function_to_string(x{i});
			out = [out,value];
		end
		out = [out,'}'];
	else
		out = mat2str(x);
	end
end
