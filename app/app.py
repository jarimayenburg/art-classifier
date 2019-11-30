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
        for label in data['ic']:
            for i in range(len(labels)):
                if label == labels[i]:
                    containedlabels.append(i)
        data['ic'] = containedlabels
        print(containedlabels)

    print(training_data)
    print("Building network...")
    config.neuralnet = network.get_network(64, len(labels))
    config.labels = labels

    training_input = []
    training_output = []

    for data in training_data:
        training_input.append(data['img'])
        training_output.append(data['ic'])

    final_training_input = []
    final_training_output = []

    for i in range(len(training_input)):
        for iconvalue in training_output[i]:
            final_training_input.append(training_input[i])
            final_training_output.append(iconvalue)

    print(np.shape(final_training_input))
    print(np.shape(final_training_output))

    print("Training network...")
    history = network.train(config.neuralnet, np.array(final_training_input), np.array(final_training_output))

    app = App(__name__, specification_dir='./')
    app.add_api('swagger.yml')
    app.run(debug=True, host='0.0.0.0', use_reloader=False)

    print("Done, ready for predictions")
