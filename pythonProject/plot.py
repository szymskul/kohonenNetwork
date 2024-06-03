from matplotlib import pyplot as plt

class Plot:
	def Scatter(self, data):
		unpacked = data
		x_coords = [neuron[0] for neuron in unpacked]
		y_coords = [neuron[1] for neuron in unpacked]

		plt.scatter(x_coords, y_coords, label='Neurons')

	def dataScatter(self, data):
		_data = {
			'x': [point['x'] for point in data],
			'y': [point['y'] for point in data]}
		plt.scatter('x', 'y', data=_data)

	def Show(self):
		plt.show()

