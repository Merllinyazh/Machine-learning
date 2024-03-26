import itertools


def vc_dimension(hypothesis_set):
	
	
	n = 4
	while True:
		points = [(i, j) for i in range(n) for j in range(2)]
		shattered_sets = 0
		for combination in itertools.combinations(points, n):
			is_shattered = True
			for labeling in itertools.product([0, 1], repeat=n):
				hypotheses = [hypothesis_set(point) for point in combination]
				if set(hypotheses) != set(labeling):
					is_shattered = False
					break
			if is_shattered:
				shattered_sets += 1
			else:
				break
		if not is_shattered:
			break
		n += 1
	return n-1 if shattered_sets == 2**n else n-2


# Example 1: linear function hypothesis set
def linear_function(point):
	x, y = point
	return int(y >= x)


print(vc_dimension(linear_function))


OUTPUT:
2
