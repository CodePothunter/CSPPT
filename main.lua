local status, cunn = pcall(require, 'fbcunn')
if not status then
	status, cunn = pcall(require, 'cunn')
	if not status then
		print("ERROR: Could not find cunn or fbcunn.")
		os.exit()
	end
else
    print("Using fbcunn")
end

local stringx = require 'pl.stringx'
LookupTable = nn.LookupTable
require 'cutorch'
require 'utils/util'
require 'utils/queue'
require 'utils/heap'
require 'utils/reader'
require 'utils/metrics'
require 'nets/lstm'
require 'nets/rnn'

--[[
#########################################################
        Deep LSTM/RNN SLU implementation via torch        
                  Su Zhu, Wengong Jin
               Email: zhusu.china@gmail.com, acmgokun@gmail.com
        Speech Lab, Shanghai Jiao Tong University        
#########################################################
--]]

local cmd = torch.CmdLine() -- cmd is used for decode parameters in the NN

cmd:text('General Options:')
cmd:option('-train', '', 'training set file')
cmd:option('-valid', '', 'validation set file')
cmd:option('-test', '', 'test set file')
cmd:option('-read_model', '', 'read model from this file')
cmd:option('-print_model', '', 'print model to this file')
cmd:option('-vocab', '', 'read vocab from this file')
cmd:option('-outlabel', '', 'read output label from this file')
cmd:option('-print_vocab', '', 'print vocab to this file')
cmd:option('-trace_level', 0, 'trace level')
cmd:option('-test_only', 0, 'only test a test file using an existing model')

cmd:text('Model Options:')
cmd:option('-rnn_type', 'lstm', 'recurrent type: lstm or rnn')
cmd:option('-emb_size', 100, 'word embedding dimension')
cmd:option('-word_win_left', 0, 'number of words in the previous context window')
cmd:option('-word_win_right', 0, 'number of words in the next context window')
-- cmd:option('-layers', 1, 'number of recurrent layers')
-- cmd:option('-hidden_size', 300, 'hidden layer dimension')
cmd:option('-hidden_prototype', '200-300', 'hidden layer dimension of each hidden layer')

cmd:text('Runtime Options:')
cmd:option('-deviceId', 1, 'train model on ith gpu')
cmd:option('-random_seed', 7, 'set initial random seed')

cmd:text('Training Options:')
cmd:option('-alpha', 0.08, 'initial learning rate')
cmd:option('-beta', 0, 'regularization constant')
cmd:option('-momentum', 0, 'momentum')
cmd:option('-dropout', 0, 'dropout rate at each non-recurrent layer')
cmd:option('-batch_size', 32, 'number of minibatch')
cmd:option('-bptt', 10, 'back propagation through time')
cmd:option('-alpha_decay', 0.6, 'alpha *= alpha_decay if no improvement on validation set')
cmd:option('-init_weight', 0.1, 'all weights will be set to [-init_weight, init_weight] during initialization')
cmd:option('-max_norm', 50, 'threshold of gradient clipping (2-norm)')
cmd:option('-max_epoch', 20, 'max number of epoch')
cmd:option('-min_improvement', 1.01, 'start learning rate decay when improvement less than threshold')
cmd:option('-shuffle', 1, 'whether to shuffle data before each epoch')

-- parse input params
local options = cmd:parse(arg)
-- print("OK")
if options.deviceId > 0 then
	cutorch.setDevice(options.deviceId)
else options.deviceId = chooseGPU()
	cutorch.setDevice(options.deviceId)
	local device_params = cutorch.getDeviceProperties(options.deviceId)
	local computability = device_params.major * 10 + device_params.minor
	if computability < 35 and options.trace_level > 0 then
		print("WARNING: fbcunn requires GPU with cuda computability >= 3.5, falling back to cunn.")
	else
		use_fbcunn = true
		LookupTable = nn.LookupTableGPU
	end
end
random_seed(options.random_seed)
--print(torch.zeros(1, 1):cuda():uniform())

local vocab = Vocab()
if options.vocab == '' then
    if options.outlabel == '' then
	    vocab:build_vocab(options.train,true)
    else
        vocab:build_vocab(options.train,false)
        vocab:build_vocab_output(options.outlabel)
    end
else
    if options.outlabel == '' then
	vocab:build_vocab(options.vocab,true)
    else
        vocab:build_vocab(options.vocab,false)
        vocab:build_vocab_output(options.outlabel)
    end
end

options.vocab_size = vocab:vocab_size()
--print(options.vocab_size["input"] .. ' ' .. options.vocab_size["output"])
options.vocab = vocab
if options.print_vocab ~= '' then
	options.vocab:save(options.print_vocab,options.print_vocab .. '.label')
end
if options.trace_level > 0 then
	cmd:log('/dev/null', options)
	io.stdout:flush()
end

local hidden_prototype = stringx.split(options.hidden_prototype, '-')
options.layers = #hidden_prototype
options.hidden_size = {}
for i = 1, #hidden_prototype do
    options.hidden_size[i] = tonumber(hidden_prototype[i])
end

local slu
if options.rnn_type == 'lstm' then
	slu = LSTM(options)
else
	slu = RNN(options)
end
slu:init(options.read_model)

local start_time = torch.tic()
local alpha_decay = false

if options.train ~= '' and options.valid ~= '' and options.test_only == 0 then
    local result = {}
    local best_f1 = -1
    local len, best_ce, res_valid = slu:evaluate(options.valid, options.valid .. '.iter0')
    local test_len, test_ce, res_test = slu:evaluate(options.test, options.test .. '.iter0')
    print('Epoch 0 validation result: words = ' .. len .. ', CE = ' .. best_ce .. ', F1 = ' .. string.format('%.4f',res_valid['F1']))
    print('Epoch 0 test result: words = ' .. test_len .. ', CE = ' .. test_ce .. ', F1 = ' .. string.format('%.4f',res_test['F1'])) 
    io.stdout:flush()
    slu:save_model(options.print_model)
    local print_model = options.print_model
    for iter = 1, options.max_epoch do
        print('Start training epoch ' .. iter .. ', learning rate: ' .. string.format("%.3f", options.alpha))
	if options.shuffle == 1 then
	    os.execute('./utils/shuf.sh ' .. options.train .. ' ' .. options.random_seed)
	end
	slu:train_one_epoch(options.train)
	len, valid_ce, res_valid = slu:evaluate(options.valid, options.valid .. '.iter' .. iter)
	test_len, test_ce, res_test = slu:evaluate(options.test, options.test .. '.iter' .. iter)
	print('Epoch ' .. iter .. ' validation result: tested words = ' .. len .. ', CE = ' .. valid_ce .. ', F1 = ' .. string.format('%.4f',res_valid['F1']))
	print('Epoch ' .. iter .. ' test result: words = ' .. test_len .. ', CE = ' .. test_ce .. ', F1 = ' .. string.format('%.4f',res_test['F1']))
		--[[if alpha_decay or best_ce / valid_ce < options.min_improvement then
			options.alpha = options.alpha * options.alpha_decay
			alpha_decay = true
		elseif iter == options.max_epoch then
			options.max_epoch = options.max_epoch + 1
		end--]]
		--[[if best_ce < valid_ce then
			slu:restore(print_model)
            print('Model is restored to previous epoch.')
		else 
			best_ce = valid_ce
            print_model = options.print_model .. '.iter' .. iter
			slu:save_model(print_model)
		end--]]
        if best_f1 < res_valid['F1'] then
            slu:save_model(print_model)
            best_f1 = res_valid['F1']
            print('NEW BEST: epoch ' .. iter .. ' best valid F1 ' .. res_valid['F1'] .. ' test F1 ' .. res_test['F1'])
            result['vf1'], result['vp'], result['vr'], result['vce'] = res_valid['F1'], res_valid['precision'], res_valid['recall'], valid_ce
            result['tf1'], result['tp'], result['tr'], result['tce'] = res_test['F1'], res_test['precision'], res_test['recall'], test_ce
            result['iter'] = iter
        end
	io.stdout:flush()
    end
    local elapsed_time = torch.toc(start_time) / 60
    print('Training finished, elapsed time = ' .. string.format('%.1f', elapsed_time))
    print('BEST RESULT: epoch ' .. result['iter'] .. ' best valid F1 ' .. result['vf1'] .. ' test F1 ' .. result['tf1'])
else
    local ctn = "1\n"
    while ctn == "1\n" do
        print("Please input the text:")
        os.execute("python test/convert2list.py 2> test/error_test.txt")
        if options.test_only == 1 and options.read_model ~= '' then
             local test_len, test_ce, res_test = slu:evaluate(options.test, options.test .. '.result')
            print("Your result")
        end
        os.execute("python test/output.py")
        ctn = io.read(2)
    end
end
