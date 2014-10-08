from cvxopt.solvers import qp
from cvxopt.base import matrix
from functools import partial

import numpy, pylab, random, math

def K_linear(p1, p2):
	scalar = 1
	for i in range(0, len(p1)):
		scalar += p1[i] * p2[i]
	return scalar

def K_polynomial(p1, p2, p):
	return pow(K_linear(p1, p2), p)

def build_p_matrix(points, classes, K):
	N = len(points)
	P = matrix(0.0, (N, N))
	for i in range(0, N):
		for j in range(0, N):
			P[i, j] = classes[i] * classes[j] * K(points[i], points[j])
	return P

def build_diagonal_matrix(diagonal_values, other_values, N):
	M = matrix(other_values, (N, N))
	for i in range(0, N):
		M[i, i] = diagonal_values
	return M

def close_to_zero(value):
	tresh = 0.00001
	return value >= -tresh and value <= tresh

def non_zero_alpha_indices(alpha):
	indices = []
	N = len(alpha)
	for i in range(0, N):
		if not close_to_zero(alpha[i]):
			indices.append(i)
	return indices

def indicator(point, alphas, points, classes, K):
	N = len(alphas)
	value = 0
	for i in range(0, N):
		value += alphas[i] * classes[i] * K(point, points[i])
	return value

def generate_data(N):
	classA = [(random.normalvariate(-3, 0.5), random.normalvariate(-0.5, 0.5), 1.0) for i in range(N)] + [(random.normalvariate(3, 0.5), random.normalvariate(-0.5, 0.5), 1.0) for i in range(N)]
	# classA = [(random.normalvariate(-3, 0.5), random.normalvariate(0.5, 1), 1.0) for i in range(N)] + [(random.normalvariate(-3, 0.5), random.normalvariate(0.5, 1), 1.0) for i in range(N)]
	classB = [(random.normalvariate(0.0, 0.5), random.normalvariate(-0.5, 0.5), -1.0) for i in range(N)] + [(random.normalvariate(-6, 0.5), random.normalvariate(-0.5, 0.5), -1.0) for i in range(N)]
	data = classA + classB
	random.shuffle(data)
	return data

def is_class_a(point):
	return point[len(point) - 1] == 1

def is_class_b(point):
	return point[len(point) - 1] == -1

def plot_data(data):
	classA = filter(is_class_a, data)
	classB = filter(is_class_b, data)
	pylab.hold(True)
	pylab.plot([p[0] for p in classA], [p[1] for p in classA], 'bo')
	pylab.plot([p[0] for p in classB], [p[1] for p in classB], 'ro')

def plot_decision_boundary(ind):
	x_range = numpy.arange(-10, 10, 0.05)
	y_range = numpy.arange(-10, 10, 0.05)
	grid = matrix([[ind([x, y]) for y in y_range] for x in x_range])
	pylab.contour(x_range, y_range, grid, (-1.0, 0.0, 1.0), colors=('red', 'black', 'blue'), linewidths=(1, 3, 1))

def run(points, classes, K):
	N = len(points)
	q = matrix(-1.0, (N, 1))
	h = matrix(0.0, (N, 1))
	G = build_diagonal_matrix(-1.0, 0.0, N)
	P = build_p_matrix(points, classes, K)

	r = qp(P, q, G, h)
	alphas = list(r['x'])

	picked = non_zero_alpha_indices(alphas)
	picked_p = []
	picked_a = []
	picked_c = []
	for i in picked:
		picked_p.append(points[i])
		picked_a.append(alphas[i])
		picked_c.append(classes[i])

	ind = partial(indicator, alphas=picked_a, points=picked_p, classes=picked_c, K=K)
	plot_decision_boundary(ind)

data = generate_data(10)
points = map(lambda d: d[:-1], data)
classes = map(lambda d: d[-1], data)
plot_data(data)
run(points, classes, partial(K_polynomial, p=3))
pylab.show()
