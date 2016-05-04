require 'luarocks.loader'
require 'nngraph'
require 'nn'
require 'rnn/LinearNoBias'

local TRAIN_LOG_WORDS = 100000

local RNN = torch.class('RNN')

function RNN:__init(options)
	self.options = options
end

-- RNN activation function
function RNN:RNN(input, prev_h, hidden_size)
    local w_h2h = nn.LinearNoBias(hidden_size, hidden_size) 
	--local w_h2h = nn.Linear(hidden_size, hidden_size) 
	local input_hidden = nn.CAddTable()({input, w_h2h(prev_h)})
	return nn.Sigmoid()(input_hidden)
end

function RNN:build_net()
	local input = nn.Identity()()
	local output_label = nn.Identity()()
	local prev_state = nn.Identity()() -- saves hidden states at all layers, hidden state includes cell activation and hidden state activation
	local next_state = {}

    local word_emb = nn.LookupTable(self.options.vocab_size['input'], self.options.emb_size)(input)
    local word_emb_input = nn.Reshape((self.options.word_win_left + 1 + self.options.word_win_right) * self.options.emb_size)(word_emb)
	local net = {[0] = nn.Linear((self.options.word_win_left + 1 + self.options.word_win_right) * self.options.emb_size, self.options.hidden_size[1])(word_emb_input)} -- net[i] saves the output of the i-th layer
	if self.options.layers > 1 then
		local prev_split = {prev_state:split(self.options.layers)}
        local prev_hidden = prev_split[1]
        local dropped_input = nn.Dropout(self.options.dropout)(net[0])
        local next_hidden = self:RNN(dropped_input, prev_hidden, self.options.hidden_size[1])
        table.insert(next_state, next_hidden)
        net[1] = next_hidden
		for i = 2, self.options.layers do
			local prev_hidden = prev_split[i]
			local dropped_input = nn.Dropout(self.options.dropout)(nn.Linear(self.options.hidden_size[i-1], self.options.hidden_size[i])(net[i - 1]))
			local next_hidden = self:RNN(dropped_input, prev_hidden, self.options.hidden_size[i])
			table.insert(next_state, next_hidden)
			net[i] = next_hidden
		end
	else
		local dropped_input = nn.Dropout(self.options.dropout)(net[0])
		local next_hidden = self:RNN(dropped_input, prev_state, self.options.hidden_size[1])
		next_state = next_hidden
		net[1] = next_hidden
	end

	local dropped_hidden = nn.Dropout(self.options.dropout)(net[self.options.layers])
    local output = nn.Linear(self.options.hidden_size[self.options.layers], self.options.vocab_size['output'])(dropped_hidden)
    local log_prob = nn.LogSoftMax()(output)
    log_prob:annotate({["name"]="prob"})
	local classifier = nn.ClassNLLCriterion()
	classifier.sizeAverage = false
	local err = classifier({log_prob, output_label})
	local model = nn.gModule({input, output_label, prev_state}, {err, nn.Identity()(next_state)})
	model:getParameters():uniform(-self.options.init_weight, self.options.init_weight)
	model = transfer2gpu(model)
	return model
end

function RNN:init(input_model)
	if input_model == '' then
		self.core_model = self:build_net()
	else
		self:load_model(input_model)
	end

    self.params, self.grads = self.core_model:getParameters()

	self.history = {}
	self.tmp_hist = {}
	if self.options.layers > 1 then
		for i = 0, self.options.bptt do
			self.history[i] = {}
			self.tmp_hist[i] = {}
			for j = 1, self.options.layers do
				self.history[i][j] = transfer2gpu(torch.zeros(self.options.batch_size, self.options.hidden_size[j])) -- torch tensor is row major, thus batch is set to row
				self.tmp_hist[i][j] = transfer2gpu(torch.zeros(self.options.batch_size, self.options.hidden_size[j])) -- torch tensor is row major, thus batch is set to row
			end
		end
		self.grad_h = {}
		for i = 1, self.options.layers do
			self.grad_h[i] = transfer2gpu(torch.zeros(self.options.batch_size, self.options.hidden_size[i]))
		end
		self.last_history = {}
		for i = 1, self.options.layers do
			self.last_history[i] = transfer2gpu(torch.zeros(1, self.options.hidden_size[i])) -- for testing, minibatch = 1
		end
	else
		self.grad_h = transfer2gpu(torch.zeros(self.options.batch_size, self.options.hidden_size[1]))
		for i = 0, self.options.bptt do
			self.history[i] = transfer2gpu(torch.zeros(self.options.batch_size, self.options.hidden_size[1])) -- torch tensor is row major, thus batch is set to row
			self.tmp_hist[i] = transfer2gpu(torch.zeros(self.options.batch_size, self.options.hidden_size[1])) -- torch tensor is row major, thus batch is set to row
		end
		self.last_history = transfer2gpu(torch.zeros(1, self.options.hidden_size[1])) -- for testing, minibatch = 1
	end
	self.err = transfer2gpu(torch.zeros(1))
	
    self.models = make_recurrent(self.core_model, self.options.bptt)
end

function RNN:forward_training()
	local len = 0
	local err = 0
    --local n_step = self.cur_batch:size()[1]
	replace(self.history[0], self.history[self.options.bptt])
    --reset(self.history[0])
	for i = 1, self.options.bptt do
		local input = self.cur_batch[i]
	    local output_label = self.cur_label[i]
        replace(self.tmp_hist[i - 1], self.history[i - 1])
        for j = 1, self.options.batch_size do
            if self.cur_pos[i][j] == 1 then
                if self.options.layers > 1 then
                    for k = 1, self.options.layers do
                        self.tmp_hist[i - 1][k][j]:zero()
                    end
                else
                    self.tmp_hist[i - 1][j]:zero()
                end
            end
        end
		err, self.history[i] = unpack(self.models[i]:forward({input, output_label, self.tmp_hist[i - 1]}))
        self.err = self.err:add(err)
	end
end

function RNN:backward()
    local n_step = self.cur_batch:size()[1]
	self.grads:mul(self.options.momentum / (-self.options.alpha))
	reset(self.grad_h)

	local derr = transfer2gpu(torch.ones(1))
	for i = n_step, 1, -1 do --self.options.bptt
		local input = self.cur_batch[i]
	    local output_label = self.cur_label[i]
		local grad_h = self.models[i]:backward({input, output_label, self.tmp_hist[i - 1]}, {derr, self.grad_h})[3] -- you can check the returned value to find this index
		replace(self.grad_h, grad_h)
		for j = 1, self.options.batch_size do
			if self.cur_pos[i][j] == 1 then
				if self.options.layers > 1 then
					for k = 1, self.options.layers do
						self.grad_h[k][j]:zero() -- clear the gradient at the end of a sentence
					end
				else
					self.grad_h[j]:zero()
				end
			end
		end
	end

    self.grads:mul(-self.options.alpha)
    --[[local grad_norm = self.grads:norm()
    if grad_norm > self.options.max_norm then
        self.grads:mul(self.options.max_norm / grad_norm) --clip gradient
    end--]]
    if self.options.beta > 0 then
        self.params:mul(1 - self.options.beta)
    end
    self.params:add(self.grads)
end

function RNN:forward_testing(probs)
    local beg_time = torch.tic()
	local err = 0
    local n_step = self.cur_batch:size()[1]
    reset(self.last_history)
    self.init_time = self.init_time + torch.toc(beg_time)
	for i = 1, n_step do
        beg_time = torch.tic()
		local input = self.cur_batch[i]
        local output_label = self.cur_label[i]
        self.t1 = self.t1 + torch.toc(beg_time)
        beg_time = torch.tic()
		local tmp_history
        err, tmp_history = unpack(self.models[1]:forward({input, output_label, self.last_history}))
        replace(self.last_history, tmp_history)
        self.t2 = self.t2 + torch.toc(beg_time)
        beg_time = torch.tic()
        --print(self.options.vocab:inv_get_input(input[1])..' '..self.options.vocab:inv_get_output(output_label[1]))
        self.err = self.err:add(err)
        self.t3 = self.t3 + torch.toc(beg_time)
        beg_time = torch.tic()
        self.forward_time = self.forward_time + torch.toc(beg_time)
        local evaluate_begin_time = torch.tic()
        local prob = probs[1].output
        if i == n_step then
            self.F1:get_batch(prob, input, output_label, 1)
        else
            self.F1:get_batch(prob, input, output_label, 0)
        end
        self.evaluate_time = self.evaluate_time + torch.toc(evaluate_begin_time)
	end
end

function RNN:train_one_epoch(train)
	self.reader = DataReader(train, self.options.batch_size, self.options.vocab, self.options.word_win_left, self.options.word_win_right)
	reset(self.history[self.options.bptt]) -- copied to self.history[0] in forward()
	self.grads:zero()
    self.err:zero()
	local len = 0
    local ce = 0
	local begin_time = torch.tic()
    local read_time, trf_time, forward_time, bp_time = 0, 0, 0, 0
	while true do
        local beg_time = torch.tic()
		self.cur_batch, self.cur_label, self.cur_pos = self.reader:get_batch_4train(self.options.bptt) -- the last word in each batch will become the first word in next batch
        read_time = read_time + torch.toc(beg_time)
		if self.cur_batch == nil then
			break
		end
		len = len + self.options.bptt * self.options.batch_size --count_words(self.options.vocab, self.cur_batch)
        beg_time = torch.tic()
		self.cur_batch = transfer2gpu(self.cur_batch)
		self.cur_label = transfer2gpu(self.cur_label)
        trf_time = trf_time + torch.toc(beg_time)
        beg_time = torch.tic()
		self:forward_training()
        forward_time = forward_time + torch.toc(beg_time)
        beg_time = torch.tic()
		self:backward()
        bp_time = bp_time + torch.toc(beg_time)
	end
    ce = self.err[1]
    print(read_time..' '..trf_time..' '..forward_time..' '..bp_time)
    local elapsed_time = torch.toc(begin_time) / 60
    print('trained words = ' .. len .. ', CE = ' .. string.format('%.3f', ce / len) .. ', elapsed time = ' .. string.format('%.1f', elapsed_time) .. ' mins.')
    len = 0
    ce = 0
    io.stdout:flush()
    collectgarbage()
end

function RNN:evaluate(data, outputfile)
    if outputfile then
        outputfile = io.open(outputfile, 'w')
    end
    self.F1 = F1(self.options.vocab, self.options.batch_size, outputfile, self.options.word_win_left) 
    local ce = 0
	local len = 0
    local probs = {}
    for i = 1, self.options.bptt do
        probs[i] = find_module(self.models[i], "prob")
    end

    local begin_time = torch.tic()
    local read_time, trf_time, forward_time = 0, 0, 0
    self.init_time, self.forward_time, self.evaluate_time = 0, 0, 0
    self.t1, self.t2, self.t3 = 0, 0, 0
	self.reader = DataReader(data, self.options.batch_size, self.options.vocab, self.options.word_win_left, self.options.word_win_right)
	--reset(self.history[self.options.bptt]) -- copied to self.history[0] in forward()
    self.err:zero()
	disable_dropout(self.models)
	while true do
        local beg_time = torch.tic()
		self.cur_batch, self.cur_label = self.reader:get_batch_4test()
		if self.cur_batch == nil then
			break
		end
        read_time = read_time + torch.toc(beg_time)
        beg_time = torch.tic()
		len = len + self.cur_batch:size()[1] --count_words(self.options.vocab, self.cur_batch)
		self.cur_batch = transfer2gpu(self.cur_batch)
		self.cur_label = transfer2gpu(self.cur_label)
        trf_time = trf_time + torch.toc(beg_time)
        beg_time = torch.tic()
		self:forward_testing(probs, true)
        forward_time = forward_time + torch.toc(beg_time)
	end
    ce = self.err[1]
    print(read_time..' '..trf_time..' '..forward_time)
    print('in forward: '..self.init_time..' '..self.forward_time..' '..self.evaluate_time)
    print('in forward: '..self.t1..' '..self.t2..' '..self.t3)
	enable_dropout(self.models)
    return len, ce/len, self.F1:get_metric()
end

function RNN:load_model(input_file)
	self.core_model = torch.load(input_file)
	self.core_model = transfer2gpu(self.core_model)
end

function RNN:restore(model)
	self:load_model(model)
	--self.core_model = transfer2gpu(self.core_model)
    self.params, self.grads = self.core_model:getParameters()
	self.models = make_recurrent(self.core_model, self.options.bptt)
	collectgarbage()
end

function RNN:save_model(output_file)
	torch.save(output_file, self.core_model)
end
