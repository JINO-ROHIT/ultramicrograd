from umgrad import Tensor

x = Tensor([1, 2, 3], requires_grad=True)
y = Tensor([4, 5, 6], requires_grad=True)
m = x.mul(y)
m.backward()

print(x, y, m)
print(x.grad, y.grad, m.grad)