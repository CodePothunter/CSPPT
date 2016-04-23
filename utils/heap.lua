local Heap = torch.class('Heap')

function Heap:down(i)
	while i * 2 <= self.size do
		local j = i * 2
		if j + 1 <= self.size and self.val[self.idx[j]] > self.val[self.idx[j + 1]] then
			j = j + 1
		end
		if self.val[self.idx[j]] < self.val[self.idx[i]] then
			self.idx[i], self.idx[j] = self.idx[j], self.idx[i]
			i = j
		else
			break
		end
	end
end

function Heap:__init(val)
	assert(type(val) == 'table')
	self.size = 0 
	self.idx = {}
	if val == nil then
		self.val = {}
		return
	end
	self.val = clone(val)
	for key in pairs(self.val) do
		self.size = self.size + 1
		self.idx[self.size] = key
		--print(key)
	end
	for i = math.floor(self.size / 2), 1, -1 do
		self:down(i)
	end
end

function Heap:pop()
	if self.size == 0 then
		return nil
	end
	local ans = self.idx[1]
	self.idx[1] = self.idx[self.size]
	self.size = self.size - 1
	self:down(1)
	return ans, self.val[ans]
end

function Heap:push(id, val)
	self.size = self.size + 1
	self.idx[self.size] = id
	self.val[id] = val
	local i = self.size
	while i > 1 do
		local j = math.floor(i / 2)
		if self.val[self.idx[j]] > self.val[self.idx[i]] then
			self.idx[j], self.idx[i] = self.idx[i], self.idx[j]
			i = j 
		else
			break
		end
	end
end
