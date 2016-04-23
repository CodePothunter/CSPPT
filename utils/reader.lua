local stringx = require 'pl.stringx'

local Vocab = torch.class('Vocab')
local DataReader = torch.class('DataReader')

function Vocab:__init()
    self.input = {}
	self.input.vocab = {['<padding>']=1, ['<null>']=2, ['<unk>']=3}
	self.input.inv_map = {'<padding>', '<null>', '<unk>'}
	self.input.size = #self.input.inv_map
    self.output = {}
    self.output.vocab = {}
	self.output.inv_map = {}
    self.output.size = 0
    self.unk = {['<unk>']=0, ['<UNK>']=0,}
end

function Vocab:add_input(word)
    if self.input.vocab[word] == nil then
	self.input.size = self.input.size + 1
	self.input.vocab[word] = self.input.size
	self.input.inv_map[self.input.size] = word
    end
end

function Vocab:add_output(label)
    if self.output.vocab[label] == nil then
	self.output.size = self.output.size + 1
	self.output.vocab[label] = self.output.size
	self.output.inv_map[self.output.size] = label
   end
end

--function Vocab:is_eos(x)
--	return x == self:get_input('<null>')
--end

function Vocab:is_null(x)
    return x == self:get_input('<null>')
end

function Vocab:get_padding()
    return self.input.vocab['<padding>']
end

function Vocab:get_input(word)
    if self.input.vocab[word] == nil then
	return self.input.vocab['<unk>']
    end
    return self.input.vocab[word]
end

function Vocab:get_output(word)
    if self.output.vocab[word] == nil then
	return self.output.vocab['<unk>']
    end
    return self.output.vocab[word]
end

function Vocab:inv_get_input(idx)
    if idx > self.input.size then
	return '<unk>'
    end
    return self.input.inv_map[idx]
end

function Vocab:inv_get_output(idx)
    if idx > self.output.size then
        return 'E'
    end
    return self.output.inv_map[idx]
end

function Vocab:vocab_size()
    return {['input']=self.input.size, ['output']=self.output.size}
end

function Vocab:save(input_file, output_file)
    local file = torch.DiskFile(input_file, 'w')
    for i,v in ipairs(self.input.inv_map) do
    	file:writeString(v .. '\n')
    end
    file:close()
    local file = torch.DiskFile(output_file, 'w')
    for i,v in ipairs(self.output.inv_map) do
        file:writeString(v .. '\n')
    end
    file:close()
end

function Vocab:build_vocab(input_file, label_or_not)
    local input, output = {}, {}
    local file = torch.DiskFile(input_file, 'r')
    file:quiet()
    while true do
        if file:hasError() then
            print(input_file .. " has error before processing!")
            break
        end
        local feats, labels = readline(file)
	if file:hasError() then
            break
        end
        --print(table.concat(feats,' ')) 
        --print(table.concat(labels,' '))
        if #feats ~= #labels then
            os.exit()
        end
        for i = 1, #feats do
            if input[feats[i]] == nil then
                input[feats[i]] = 1
            else
                input[feats[i]] = input[feats[i]] + 1
            end
            if label_or_not ~= false then
                if output[labels[i]] == nil then
                    output[labels[i]] = 1
                else
                    output[labels[i]] = output[labels[i]] + 1
                end
            end
        end
    end
    file:close()
    for word in pairs(input) do
        if input[word] ~= 0 and self.unk[word] == nil then
            self:add_input(word)
        end
    end
    if label_or_not ~= false then
        for label in pairs(output) do
            self:add_output(label)
        end
    end
end

function Vocab:build_vocab_output(input_file)
    local input, output = {}, {}
	local file = torch.DiskFile(input_file, 'r')
	file:quiet()
	while true do
		local data = readline1(file)
		if file:hasError() then
			break
		end
        self:add_output(data[1])
	end
end


function DataReader:__init(input_file, batch_size, vocab, word_win_left, word_win_right)
    assert(word_win_left>=0)
    assert(word_win_right>=0)
    self.word_win_left = word_win_left
    self.word_win_right = word_win_right

    self.input_file = torch.DiskFile(input_file, 'r')
    self.input_file:quiet()
    self.batch_size = batch_size
    self.vocab = vocab
    self.batch = {}
    self.batch_label = {}
    self.batch_pos = {}
    for i = 1, self.batch_size do
        self.batch[i] = {}
        self.batch_label[i] = {}
        self.batch_pos[i] = {}
    end
    self.streams = {}
    self.streams_label = {}
    self.streams_pos = {}
    for i = 1, self.batch_size do
        self.streams[i] = Queue()
        self.streams_label[i] = Queue()
        self.streams_pos[i] = Queue()
    end
end

function DataReader:get_stream(id)
    if self.streams[id]:is_empty() then
        local feats,labels = readline(self.input_file)
        if self.input_file:hasError() then
            local tmp = {}
            for i = 1, self.word_win_left + 1 + self.word_win_right do
                tmp[i] = self.vocab:get_input('<null>')
            end
            return tmp, self.vocab:get_output('O'), 1 
        end
        local tmp = {}
        for i = 1, self.word_win_left do
            tmp[i] = self.vocab:get_padding()  -- PADDING index in vocab
        end
        for i = 1, #feats do
            tmp[self.word_win_left + i] = self.vocab:get_input(feats[i])
            self.streams_label[id]:push(self.vocab:get_output(labels[i]))
            self.streams_pos[id]:push(i)
        end
        for i = 1, self.word_win_right do
             tmp[self.word_win_left + #feats + i] = self.vocab:get_padding()  -- PADDING index in vocab
        end
        for i = 1, #feats do
            local item = {}
            for k = -self.word_win_left, self.word_win_right do
                item[self.word_win_left + k + 1] = tmp[self.word_win_left + i + k]
            end
            self.streams[id]:push(item)
        end
    end
    return self.streams[id]:pop(), self.streams_label[id]:pop(), self.streams_pos[id]:pop()
end

function DataReader:get_batch_4train(length)
    local goon = false
    for i = 1, self.batch_size do
        for j = 1, length do
            self.batch[i][j], self.batch_label[i][j], self.batch_pos[i][j] = self:get_stream(i)
            if not self.vocab:is_null(self.batch[i][j][1]) then -- context
                goon = true
            end
        end
    end
    
    if goon then
        for i = 1, self.batch_size do
            self.batch[i][length + 1] = nil
            self.batch_label[i][length + 1] = nil
            self.batch_pos[i][length + 1] = nil
        end
         
        return torch.Tensor(self.batch):transpose(1,2), torch.Tensor(self.batch_label):t(), torch.Tensor(self.batch_pos):t() -- transposed to length * batch (needed during forward())
    else
        self.input_file:close()
        return nil, nil, nil
    end
end

function DataReader:get_batch_4test()
    local batch_size = 1
    local batch = {}
    local batch_label = {}
    local batch_context = {}
    for i = 1, batch_size do
        batch[i] = {}
        for j = 1, self.word_win_left do
            batch[i][j] = self.vocab:get_padding()  -- PADDING index in vocab
        end
        batch_label[i] = {}
        batch_context[i] = {}
    end
    local feats,labels = readline(self.input_file)
    --print (self.input_file)   
    --print (#feats ..' '.. #labels)
    if self.input_file:hasError() then
        self.input_file:close()
        return nil, nil
    end
    for j = 1, #feats do
	for i = 1, batch_size do
            batch[i][self.word_win_left + j], batch_label[i][j] = self.vocab:get_input(feats[j]), self.vocab:get_output(labels[j])
	end
    end

    for i = 1, batch_size do
        for j = 1, self.word_win_right do
            batch[i][self.word_win_left + #feats + j] = self.vocab:get_padding()
        end
    end

    for j = 1, #feats do
        for i = 1, batch_size do
            batch_context[i][j] = {}
            for k = -self.word_win_left, self.word_win_right do
                batch_context[i][j][self.word_win_left + k + 1] = batch[i][self.word_win_left + j + k]
            end
        end
    end

    return torch.Tensor(batch_context):transpose(1,2), torch.Tensor(batch_label):t() -- transposed to length * batch (needed during forward())
end

