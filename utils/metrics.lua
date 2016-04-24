local stringx = require 'pl.stringx'

local F1 = torch.class('F1')

function F1:__init(vocab, batch_size, outputfile, word_win_left)
    self.TP = 0
    self.FP = 0
    self.FN = 0
    self.TN = 0
    self.vocab = vocab
    self.vocab_size = vocab:vocab_size()
    self.label_cache, self.pred_cache, self.input_cache = {}, {}, {}
    for i = 1, batch_size do
        self.label_cache[i] = {'O'}
        self.pred_cache[i] = {'O'}
        self.input_cache[i] = {'BOS'}
    end
    self.label_chunks, self.pred_chunks = {}, {}
    self.outputfile = outputfile
    assert(word_win_left>=0)
    self.word_pos = word_win_left+1
    self.trf_time, self.cal_time, self.write_time = 0, 0, 0
end

function F1:get_metric()
    if self.TP == 0 then
        return {['precision']=0, ['recall']=0, ['F1']=0}
    end
    print(self.trf_time..'#'..self.cal_time..'#'..self.write_time)
    return {['precision']=self.TP/(self.TP+self.FP), ['recall']=self.TP/(self.TP+self.FN), ['F1']=2*self.TP/(2*self.TP+self.FP+self.FN)}
end

function F1:get_batch(probs, input, labels, pos)
    probs = probs:float()
    for i = 1, probs:size()[1] do
        if pos == 1 then
            if #self.label_cache[i] ~= 1 then
                local beg_time = torch.tic()
                self.label_cache[i][#self.label_cache[i]+1] = 'O'
                self.pred_cache[i][#self.pred_cache[i]+1] = 'O'
                self.input_cache[i][#self.input_cache[i]+1] = 'EOS'
                if self.outputfile ~= nil then
                    for k = 1, #self.input_cache[i] do
                        self.outputfile:write(self.input_cache[i][k] .. ' ' .. self.label_cache[i][k] .. ' ' .. self.pred_cache[i][k] .. '\n')
                        -- print(self.input_cache[i][k] .. ' ' .. self.label_cache[i][k] .. ' ' .. self.pred_cache[i][k] .. '\n')
                    end
                    self.outputfile:write('\n')
                end
                self.label_chunks = self:get_chunks(self.label_cache[i])
                self.pred_chunks = self:get_chunks(self.pred_cache[i])
                self.label_cache[i] = {'O'}
                self.pred_cache[i] = {'O'}
                self.input_cache[i] = {'BOS'}
                for key,value in pairs(self.pred_chunks) do
                    if string.sub(key, -1, -1) ~= 'O' and string.sub(key, -2, -1) == 'ED' then
                        if self.label_chunks[key] then 
                            self.TP = self.TP + 1
                        else
                            self.FP = self.FP + 1
                        end
                    end
                end
                for key,value in pairs(self.label_chunks) do
                    if self.pred_chunks[key] == nil and string.sub(key, -1, -1)  ~= 'O'
                      and string.sub(key, -2, -1) == 'ED' then
                        self.FN = self.FN + 1
                    end
                end
                self.write_time = self.write_time + torch.toc(beg_time)
            end
            self.outputfile:close()
            self.outputfile = io.open("test/tmp/input.list.result", "a")
        end
        self.label_cache[i][#self.label_cache[i]+1] = self.vocab:inv_get_output(labels[i])
        local beg_time = torch.tic()
        --local probs_table = {}
        --for k = 1, self.vocab_size['output'] do
        --    probs_table[k] = -probs[i][k]
        --end
        self.trf_time = self.trf_time + torch.toc(beg_time)
        beg_time = torch.tic()
        top_idx, top_prob = self:get_top(probs[i],1)
        --print(top_idx[1]..' '..top_prob[1]..';'..probs[i][1])
        self.pred_cache[i][#self.pred_cache[i]+1] = self.vocab:inv_get_output(top_idx[1])
        self.input_cache[i][#self.input_cache[i]+1] = self.vocab:inv_get_input(input[i][self.word_pos])
        self.cal_time = self.cal_time + torch.toc(beg_time)
    end
end

function F1:get_top(probs,nbest)
    --[[local heap = Heap(probs)
    local top_idx, top_prob = {}, {}
    for i = 1, nbest do
        idx, prob = heap:pop()
        if idx == nil then
            return top_idx, top_prob
        end
        top_idx[#top_idx+1], top_prob[#top_prob+1] = idx, prob
    end
    return top_idx, top_prob--]]
    local top_idx, top_prob = {[1]=0}, {[1]=-100}
    for i = 1, self.vocab_size['output'] do
        if probs[i] > top_prob[1] then
            top_idx[1] = i
            top_prob[1] = probs[i]
        end
    end
    return top_idx, top_prob
end

function F1:get_chunks(labels)
    local chunks = {}
    local start_idx, end_idx = 0, 0
    local chunkStart, chunkEnd = false, false
    local Type = 'O'
    for idx = 2, #labels-1 do
        Type = 'E'
        chunkStart, chunkEnd = idx, idx
        if labels[idx] ~= 'O' and string.sub(labels[idx],-1,-1) ~= 'E' then
            Type = 'ED'
        end
        -- Type = labels[idx]
        if chunkEnd then
            end_idx = idx
            --print (start_idx .. '-' .. end_idx .. '-' .. Type)
            chunks[start_idx..'-'..end_idx..'-'..Type] = true
        end
    end
    return chunks
end

