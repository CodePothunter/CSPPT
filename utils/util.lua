local stringx = require 'pl.stringx'

function chooseGPU()
	local cnt = cutorch.getDeviceCount()
	for i = 1, cnt do
		totalMem, freeMem = cutorch.getMemoryUsage(i)
		if freeMem / totalMem > 0.95 then
			return i
		end
	end
	print("WARNING: No Free GPU is find. Using the primary GPU card")
	return 1
end

function transfer2gpu(module)
	return module:cuda()
end

function readline(file)
    assert(file:isQuiet())
    local word_list, label_list = {},{}
    local input, output, tmp = {},{},{}
    local line = file:readString('*l')
    if file:hasError() then
        return nil
    end
    while line ~= '' do
        tmp = stringx.split(line)
        local word,label = tmp[1],tmp[#tmp]
        --print(line..' '..word..' '..label)
        input[#input+1] = word
        output[#output+1] = label
        line = file:readString('*l')
        if file:hasError() then
            return nil
        end
    end
    return input, output 
end

function readline1(file)
    assert(file:isQuiet())
    local line = file:readString('*l') -- end-of-line character is omitted!
    if file:hasError() then
    	return nil
    end
    local data = stringx.split(line)
    return data 
end

function replace(to, from)
    if type(to) == 'table' then
    	assert (#from == #to)
	for i = 1, #from do
	    to[i]:copy(from[i])
	end
    else
	to:copy(from)
    end
end

function reset(x)
	if type(x) == 'table' then
		for i = 1, #x do
			x[i]:zero()
		end
	else
		x:zero()
	end
end

function clone(x)
	local buffer = torch.MemoryFile('rw'):binary()
	buffer:writeObject(x)
	buffer:seek(1)
	local y = buffer:readObject() -- clone via memory file
	buffer:close()
	return y
end

function count_words(vocab, input)
	local cnt = 0
	assert(type(input) == 'userdata')
	if input:dim() == 1 then
		for i = 1, input:size(1) do
			if not vocab:is_null(input[i]) then
				cnt = cnt + 1
			end
		end
	elseif input:dim() == 2 then
		for i = 2, input:size(1) do
			for j = 1, input:size(2) do
				if not vocab:is_null(input[i][j]) then
					cnt = cnt + 1
				end
			end
		end
	end
	return cnt
end
					

function make_recurrent(net, times)
	local clones = {}
	local params, grads = net:parameters() -- here use parameters() instead of getParameters() because getParameters() returns flattened tables
	if params == nil then
		params = {}
	end
	local buffer = torch.MemoryFile('w'):binary()
	buffer:writeObject(net)
	for t = 1, times do
		local reader = torch.MemoryFile(buffer:storage(), 'r'):binary()
		local clone_net = reader:readObject()
		reader:close()
		local clone_params, clone_grads = clone_net:parameters()
		for i = 1, #params do
			clone_params[i]:set(params[i])
			clone_grads[i]:set(grads[i])
		end
		clones[t] = clone_net
		collectgarbage()
	end
	buffer:close()
	return clones
end

-- Related to inner design of nngraph, confused.
function disable_dropout(node)
	if type(node) == 'table' and node.__typename == nil then
		for i = 1, #node do
			node[i]:apply(disable_dropout)
		end
		return
	end
	if string.match(node.__typename, "Dropout") then
		node.train = false
	end
end

function enable_dropout(node)
	if type(node) == 'table' and node.__typename == nil then
		for i = 1, #node do
			node[i]:apply(enable_dropout)
		end
		return
	end
	if string.match(node.__typename, "Dropout") then
		node.train = true
	end
end

function random_seed(seed)
	torch.manualSeed(seed)
	cutorch.manualSeed(seed)
	torch.zeros(1, 1):cuda():uniform()
	local rand_file = torch.DiskFile('.randfile', 'w'):binary()
	for i = 1, 100000 do
		local arr = torch.rand(100):float()
		rand_file:writeFloat(arr:storage())
	end
	rand_file:close()
end

function find_module(model, pred)
    for _, node in ipairs(model.forwardnodes) do
        if stringx.startswith(node:graphNodeName(), pred) then
	    return node.data.module
	end
    end
end

