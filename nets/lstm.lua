require 'luarocks.loader'
require 'nngraph'
require 'cunn'
require 'rnn/LinearNoBias'

local TRAIN_LOG_WORDS = 100000

local LSTM = torch.class('LSTM')

function LSTM:__init(options)
    self.options = options
end

-- lstm cell activation function
-- no peephole connection.
function LSTM:lstm(input, prev_c, prev_h, input_size, hidden_size)
    -- every call to input_hidden_sum() creates three nn.Linear() modules
    -- input, forget, output gates use different weight matrices
    -- bias is automatically included in nn.Linear()
    -- nn.Linear() is a module and nn.Linear()() is a graph node in nngraph
    local function input_hidden_sum()
        local w_i2h = nn.Linear(input_size, hidden_size)
        --local w_h2h = nn.LinearNoBias(hidden_size, hidden_size) 
        local w_h2h = nn.Linear(hidden_size, hidden_size) 
        return nn.CAddTable()({w_i2h(input), w_h2h(prev_h)}) --w_i2h(input) is a node!
    end
    local function input_hidden_cell_sum(Xcell)
        local w_i2h = nn.Linear(input_size, hidden_size)
        local w_c2h = nn.CMul(hidden_size)
        --local w_h2h = nn.LinearNoBias(hidden_size, hidden_size) 
        local w_h2h = nn.Linear(hidden_size, hidden_size) 
        return nn.CAddTable()({w_i2h(input), w_c2h(Xcell), w_h2h(prev_h)}) --w_i2h(input) is a node!
    end
    local input_gate = nn.Sigmoid()(input_hidden_cell_sum(prev_c)) --nn.Sigmoid()(input_hidden_sum())
    local forget_gate = nn.Sigmoid()(input_hidden_cell_sum(prev_c)) --nn.Sigmoid()(input_hidden_sum())
    local cell_input = nn.Tanh()(input_hidden_sum())
    local cell = nn.CAddTable()({nn.CMulTable()({input_gate, cell_input}), nn.CMulTable()({forget_gate, prev_c})})
    local output_gate = nn.Sigmoid()(input_hidden_cell_sum(cell)) --nn.Sigmoid()(input_hidden_sum())
    local hidden = nn.CMulTable()({output_gate, nn.Tanh()(cell)})
    return cell, hidden --lstm returns two nodes!
end

function LSTM:build_net()
    local input = nn.Identity()() 
    local target = nn.Identity()()
    local prev_state = nn.Identity()() -- saves hidden states at all layers, hidden state includes cell activation and hidden state activation
    local next_state = {}
    local wvec = LookupTable(self.options.vocab_size['input'], self.options.emb_size)(input)
    print ("I am OK!")
    local wvec_input = nn.Reshape((self.options.word_win_left + 1 + self.options.word_win_right) * self.options.emb_size)(wvec)
    --local net = {[0] = nn.Linear((self.options.word_win_left + 1 + self.options.word_win_right) * self.options.emb_size, self.options.hidden_size[1])(wvec_input)}
    local net = {[0] = wvec_input}
    local prev_split = {prev_state:split(2 * self.options.layers)} -- each hidden layer is split, one for cell, one for hidden, there will be two successors of nn.Identity()() if nn.Identity()() is split.
    local prev_cell = prev_split[1]
    local prev_hidden = prev_split[2]
    local dropped_input = nn.Dropout(self.options.dropout)(net[0])
    local next_cell, next_hidden = self:lstm(dropped_input, prev_cell, prev_hidden, (self.options.word_win_left + 1 + self.options.word_win_right) * self.options.emb_size, self.options.hidden_size[1])
    table.insert(next_state, next_cell)
    table.insert(next_state, next_hidden)
    net[1] = next_hidden
    if self.options.layers > 1 then
        for i = 2, self.options.layers do
            local prev_cell = prev_split[2 * i - 1]
            local prev_hidden = prev_split[2 * i]
            local dropped_input = nn.Dropout(self.options.dropout)(net[i - 1])
            local next_cell, next_hidden = self:lstm(dropped_input, prev_cell, prev_hidden, self.options.hidden_size[i-1], self.options.hidden_size[i])
            table.insert(next_state, next_cell)
            table.insert(next_state, next_hidden)
            net[i] = next_hidden
        end
    end
    
    local dropped_hidden = nn.Dropout(self.options.dropout)(net[self.options.layers])
    local output = nn.Linear(self.options.hidden_size[self.options.layers], self.options.vocab_size['output'])(dropped_hidden)
    local log_prob = nn.LogSoftMax()(output)
    log_prob:annotate({["name"] = "log_prob"})
    local classifier = nn.ClassNLLCriterion()
    classifier.sizeAverage = false
    local err = classifier({log_prob, target})
    local model = nn.gModule({input, target, prev_state}, {err, nn.Identity()(next_state)}) -- input to the network (at a certain time t) is input, prev_state and target output of the network (at a certain time t) is err and next_state. (err is not a node, but next_state is changed to a node)
    model:getParameters():uniform(-self.options.init_weight, self.options.init_weight)
    model = transfer2gpu(model)
    return model
end

function LSTM:init(input_model)
    if input_model == '' then
        print('Building network')
        self.core_model = self:build_net()
    else
        if self.options.trace_level > 0 then
        print('Loading model from ' .. input_model)
        end
        self:load_model(input_model)
    end
    --self.core_model = transfer2gpu(self.core_model)
    self.params, self.grads  = self.core_model:getParameters()
    self.models = make_recurrent(self.core_model, self.options.bptt)

    self.history = {}
    self.tmp_hist = {}
    self.grad_h = {}
    self.last_history = {}
    self.err = transfer2gpu(torch.zeros(1))
    for i = 0, self.options.bptt do
        self.history[i] = {}
        self.tmp_hist[i] = {}
        for j = 1, self.options.layers do
            self.history[i][2*j-1] = transfer2gpu(torch.zeros(self.options.batch_size, self.options.hidden_size[j])) -- torch tensor is row major, thus batch is set to row
            self.history[i][2*j] = transfer2gpu(torch.zeros(self.options.batch_size, self.options.hidden_size[j])) -- torch tensor is row major, thus batch is set to row
            self.tmp_hist[i][2*j-1] = transfer2gpu(torch.zeros(self.options.batch_size, self.options.hidden_size[j])) -- torch tensor is row major, thus batch is set to row
            self.tmp_hist[i][2*j] = transfer2gpu(torch.zeros(self.options.batch_size, self.options.hidden_size[j])) -- torch tensor is row major, thus batch is set to row
        end
    end
    for i = 1, self.options.layers do
        self.grad_h[2*i-1] = transfer2gpu(torch.zeros(self.options.batch_size, self.options.hidden_size[i]))
        self.grad_h[2*i] = transfer2gpu(torch.zeros(self.options.batch_size, self.options.hidden_size[i]))
    end
    for i = 1, self.options.layers do
        self.last_history[2*i-1] = transfer2gpu(torch.zeros(1, self.options.hidden_size[i])) -- for testing, minibatch=1
        self.last_history[2*i] = transfer2gpu(torch.zeros(1, self.options.hidden_size[i])) -- for testing, minibatch=1
    end
end

function LSTM:forward_training()
    local len = 0
    local err
    local n_step = self.cur_batch:size()[1] --self.options.bptt
    replace(self.history[0], self.history[self.options.bptt])
    --reset(self.history[0])
    for i = 1, n_step do
        local input = self.cur_batch[i]
        local output_label = self.cur_label[i]
        replace(self.tmp_hist[i - 1], self.history[i - 1])
        for j = 1, self.options.batch_size do
            if self.cur_pos[i][j] == 1 then
                for k = 1, 2 * self.options.layers do
                    self.tmp_hist[i - 1][k][j]:zero()
                end
            end
        end
        err, self.history[i] = unpack(self.models[i]:forward({input, output_label, self.tmp_hist[i - 1]}))
        self.err = self.err:add(err)
    end
end

function LSTM:backward()
    local n_step = self.cur_batch:size()[1] --self.options.bptt
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
                for k = 1, 2 * self.options.layers do
                    self.grad_h[k][j]:zero() -- clear the gradient at the end of a sentence
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

function LSTM:forward_testing(probs)
    local n_step = self.cur_batch:size()[1]
    reset(self.last_history)
    --for j = 1, self.options.batch_size do
    --    self.last_history[j]:zero()
    --end
    local err
    local flaag = 0
    for i = 1, n_step do
        local input = self.cur_batch[i]
        -- print(input)
        local output_label = self.cur_label[i]
        local xx
        err, xx = unpack(self.models[1]:forward({input, output_label, self.last_history}))
        replace(self.last_history, xx)
        self.err = self.err:add(err)
        local prob = probs[1].output
        if prob ~= nil and flaag == 0 then
            -- print(prob)
            flaag = 1
        end
        if i == n_step then
            self.F1:get_batch(prob, input, output_label, 1)
        else
            self.F1:get_batch(prob, input, output_label, 0)
        end
    end
end

function LSTM:train_one_epoch(train)
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
        --print(self.cur_batch)
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
end

function LSTM:evaluate(data, outputfile)
    if outputfile then
        outputfile = io.open(outputfile, 'w')
    end
    self.F1 = F1(self.options.vocab, self.options.batch_size, outputfile, self.options.word_win_left) 
    local ce = 0 --cross entropy
    local len = 0
    local probs = {}
    -- To find all the forward modules whose node name begins with "log_prob"
    for i = 1, self.options.bptt do
        probs[i] = find_module(self.models[i], "log_prob")
    end

    local begin_time = torch.tic()
    local read_time, trf_time, forward_time = 0, 0, 0

    self.reader = DataReader(data, self.options.batch_size, self.options.vocab, self.options.word_win_left, self.options.word_win_right)
    --reset(self.history[self.options.bptt]) -- copied to self.history[0] in forward()
    self.err:zero()
    -- What ?
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
    -- print(read_time..' '..trf_time..' '..forward_time)
    enable_dropout(self.models)
    return len, ce/len, self.F1:get_metric()
end

function LSTM:load_model(input_file)
    self.core_model = torch.load(input_file)
    self.core_model = transfer2gpu(self.core_model)
end

function LSTM:restore(model)
    self:load_model(model)
    --self.core_model = transfer2gpu(self.core_model)
    self.params, self.grads = self.core_model:getParameters()
    self.models = make_recurrent(self.core_model, self.options.bptt)
    collectgarbage()
end

function LSTM:save_model(output_file)
    torch.save(output_file, self.core_model)
end
