# Application entry point

from flask import Flask
from connexion import App
from classifier import api, network
import config
import numpy as np

# Entrypoint
if __name__ == '__main__':
    config.init()

    print("Retrieving training data...")
    training_data = network.get_training_data(64)

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

    print("Building network...")
    config.neuralnet = network.get_network(64, len(labels))
    config.labels = labels

    training_input = []
    training_output = []

    for data in training_data:
        training_input.append(data['img'])
        training_output.append(data['ic'])

    training_input = np.array(training_input)
    training_output = np.array(training_output)

    print("Training network...")
    history = network.train(config.neuralnet, training_input, training_output)

    app = App(__name__, specification_dir='./')
    app.add_api('swagger.yml')
    app.run(debug=True, host='0.0.0.0', use_reloader=False)

    print("Done, ready for predictions")
