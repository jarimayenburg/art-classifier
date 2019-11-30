# Application entry point

from flask import Flask
from connexion import App
from classifier import api, network

app = App(__name__, specification_dir='./')

# Read the Swagger specification
app.add_api('swagger.yml')

neuralnet = None

# Entrypoint
if __name__ == '__main__':
    training_data = network.get_training_data(64)
    print(training_data)
    print(len(training_data))

    # Prepare labels
    labels = list()
    for data in training_data:
        if data['ic'] not in labels:
            labels.append(data['ic'])

    for data in training_data:
        for i in range(len(labels)):
            if labels[i] == data['ic']:
                data['ic'] = i
                break

    print(training_data)
    print(len(labels))

    neuralnet = network.get_network(64, len(labels))

    training_input = list()
    training_output = list()

    for data in training_data:
        training_input.append(data['img'])
        training_output.append(data['ic'])

    history = network.train(neuralnet, training_input, training_output)

    print(history)

    app.run(debug=True, host='0.0.0.0')
