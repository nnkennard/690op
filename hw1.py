import cvxpy as cp
import numpy as np

import matplotlib.pyplot as plt


k = np.loadtxt("xy_train.csv", delimiter=",")
x = np.array(k[:,:2], dtype=np.float32)
y = np.array(k[:,2], dtype=np.int32)

p = 2
n = len(y)

beta = cp.Variable(p)
beta_0 = cp.Variable()
xis = [cp.Variable() for _ in range(n)]

C = 1
 
# Construct the problem.
objective = cp.Minimize(0.5 * cp.power(cp.norm(beta, 2), 2) + C * cp.sum(xis))
constraints = [xi_i >= 0 for xi_i in xis] + [
  (y[i] * (cp.sum(x[i]*beta) + beta_0)) >= (1 - xis[i]) for i in range(n)]
prob = cp.Problem(objective, constraints)

result = prob.solve()


w = beta.value
a = -w[0] / w[1]
xx = np.linspace(-2.5, 2.5)
yy = a * xx - (beta_0.value) / w[1]

fig, ax = plt.subplots()
ax.scatter(x[:,0], x[:,1], c=y)
plt.plot(xx, yy , linestyle='solid')
plt.show()
