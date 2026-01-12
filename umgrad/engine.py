from functools import partialmethod

class Tensor:
    def __init__(self, data, requires_grad = False):
        if isinstance(data, (int, float)):
            self.data = [data]
        else:
            self.data = data
        self.grad = None
        self.requires_grad = requires_grad
        self._ctx = None
    
    def __repr__(self):
        return f"Tensor({self.data}), requires_grad={self.requires_grad}"
    
    def _broadcast(self, other):
        """need this only for binary ops
        rules:
            start from rightmost side toward left, two dims are compactible if they are -
            1. equal
            2. one of them is 1
        """
        pass

    def backward(self, grad = None):
        if not self.requires_grad:
            return
        
        if grad is None:
            grad = [1.0] * len(self.data)
        
        if self.grad is None:
            self.grad = grad
        else:
            self.grad = [a + b for a, b in zip(self.grad, grad)] # ugly?
        
        if self._ctx is None:
            return
        
        grads = self._ctx.op.backward(self._ctx, self.grad)

        for tensor, g in zip(self._ctx.saved_tensors, grads): # topo sort is probably safer
            if tensor.requires_grad:
                tensor.backward(g)

class Context:
    def __init__(self, op):
        self.op = op
        self.saved_tensors = []
    
    def save_for_backward(self, *x):
        self.saved_tensors.extend(x)
    
    @staticmethod
    def apply(op, *x):
        ctx = Context(op)
        data = op.forward(ctx, *x)
        out = Tensor(data, requires_grad = any(_x.requires_grad for _x in x))
        out._ctx = ctx
        return out


def register(name, fn):
    setattr(Tensor, 
            name, 
            partialmethod(lambda self, other: Context.apply(fn, self, other))
    )
    
class Mul:
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return [a * b for a, b in zip(x.data, y.data)] # to-do: possibly wont work bc of broadcasting?
    
    @staticmethod
    def backward(ctx, grad):
        x, y = ctx.saved_tensors
        return [g * b for g, b in zip(grad, y.data)], [g * a for g, a in zip(grad, x.data)]
    
register("mul", Mul)

if __name__ == "__main__":
    x = Tensor([1, 2, 3], requires_grad=True)
    y = Tensor([4, 5, 6], requires_grad=True)
    z = Tensor([10], requires_grad=True)
    m = x.mul(y)

    m.backward()

    print(x, y, m)
    print(x.grad, y.grad, m.grad)

    # TO-DO make this work
    # m2 = m.mul(z)

    # print(x, y, m, m2)
    # print(m._ctx.saved_tensors)
    # print(m._ctx.op)

    # print(m2._ctx.saved_tensors)
    # print(m2._ctx.op)