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
        for label in data['ic']:
            if label not in labels:
                labels.append(label)

    for data in training_data:
        containedlabels = list()
        for label in labels:
            if label in data['ic']:
                containedlabels.append(1.0)
            else:
                containedlabels.append(0.0)
        data['ic'] = containedlabels

    print("Building network...")
    config.neuralnet = network.get_network(64, len(labels))
    config.labels = labels

    training_input = []
    training_output = []

    for data in training_data:
        training_input.append(data['img'])
        training_output.append(data['ic'])

    print("Training network...")
    history = network.train(config.neuralnet, np.array(training_input), np.array(training_output))

    app = App(__name__, specification_dir='./')
    app.add_api('swagger.yml')
    app.run(debug=True, host='0.0.0.0', use_reloader=False)

    print("Done, ready for predictions")
