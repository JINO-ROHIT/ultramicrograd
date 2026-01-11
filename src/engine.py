from functools import partialmethod

class Tensor:
    def __init__(self, data, requires_grad = False):
        self.data = data
        self.grad = 0 if requires_grad else None
        self.requires_grad = requires_grad

        self._ctx = None
    
    def __repr__(self):
        return f"Tensor({self.data}), requires_grad={self.requires_grad}"

    def backward(self, grad = None):
        if self._ctx is None:
            return

class Context:
    def __init__(self, op):
        self.op = op
        self.saved_tensors = []
    
    def save_for_backward(self, *x):
        self.saved_tensors.extend(x)
    
    @staticmethod
    def apply(op, *x):
        ctx = Context(op)
        data = op.forward(ctx, *(_x.data for _x in x))
        out = Tensor(data, requires_grad = any(_x.requires_grad for _x in x))
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
        return [a * b for a, b in zip(x, y)]
    
    @staticmethod
    def backward(ctx, x, y):
        pass

class Sum:

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return sum(x.data)
    
register("mul", Mul)
register("sum", Sum)

if __name__ == "__main__":
    x = Tensor([1, 2, 3], requires_grad=True)
    y = Tensor([4, 5, 6], requires_grad=True)
    m = x.mul(y)

    print(x, y, m)