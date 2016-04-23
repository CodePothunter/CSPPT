local Queue = torch.class('Queue')

function Queue:__init()
	self.left = 0
	self.right = -1
	self.queue = {}
end

function Queue:pop()
	local ret = self.queue[self.left]
	self.queue[self.left] = nil
	self.left = self.left + 1
	return ret
end

function Queue:push(val)
	self.right = self.right + 1
	self.queue[self.right] = val
end

function Queue:is_empty()
	return self.left > self.right
end
