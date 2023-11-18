import numpy as np

create_array = lambda array: np.array(array)

test = create_array([1,2,3,4,5,6,7,8,9])

test = np.reshape(test, (3, 3))
print('\n'.join(' '.join(map(str, row)) for row in test))

