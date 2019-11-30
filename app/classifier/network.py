import tensorflow as tf
import numpy as np
import requests
from PIL import Image
from io import BytesIO
from SPARQLWrapper import SPARQLWrapper, JSON
from threading import Thread, Lock, Semaphore

def make_square(im, fill_color=(0, 0, 0)):
    """
    Makes the given PIL Image object a square one, with the
    extra pixels filled in with the provided fill_color
    """
    x, y = im.size
    size = max(x, y)
    new_im = Image.new('RGB', (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    return new_im

def resize(image, size):
    """
    Resizes the given PIL Image object to a given (square) size. Fills
    in the spaces of rectangular images with black
    """
    image_matrix = make_square(image)
    image_matrix = image_matrix.resize((size, size), Image.NONE)
    return image_matrix


def get_training_data_raw():
    # result["img"]["value"]
    # result["ic"]["value"]
    sparql = SPARQLWrapper("https://api.data.netwerkdigitaalerfgoed.nl/datasets/hackalod/RM-PublicDomainImages/services/RM-PublicDomainImages/sparql")
    sparql.setQuery("""
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX dc: <http://purl.org/dc/elements/1.1/>
    SELECT distinct ?img ?ic WHERE {
    ?src <http://www.europeana.eu/schemas/edm/isShownBy> ?img .
    ?src <http://www.europeana.eu/schemas/edm/isShownAt> ?sub .

    ?sub dc:subject ?ic .
    FILTER(CONTAINS(str(?ic), "http://iconclass.org")) .
    } LIMIT 500
    """)

    sparql.setReturnFormat(JSON)
    return sparql.query().convert()

def get_single_data_instance(img_size, datapoints, data, lock, counters):
    localentries = list()
    localfailed = 0
    localok = 0

    for datapoint in datapoints:
        # Get image URL
        img = datapoint['img']['value']
        img = img[:-3]
        img = img + '=s' + str(img_size)

        # Get actual image
        tries = 0
        while tries < 4:
            try:
                response = requests.get(img, timeout=20)
                localok += 1
                break
            except requests.RequestException:
                tries += 1

        if(tries >= 4):
            localfailed += 1
            print("Skipping entry with URL: " + img)

        img = Image.open(BytesIO(response.content))
        img = resize(img, img_size)
        img = np.array(img)

        ic = datapoint['ic']['value']
        entry = {'img': img, 'ic': ic}

        localentries.append(entry)

    lock.acquire()
    counters['ok'] += localok
    counters['fail'] += localfailed
    for localentry in localentries:
        data.append(localentry)
    lock.release()

def divide_chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

def get_training_data(img_size):
    data = list()
    print("Getting raw training data JSON")
    raw_data = get_training_data_raw()['results']['bindings']

    # For progress tracking
    threads = list()
    mutex = Lock()
    threadcount = 256
    counters = {'ok': 0, 'fail': 0}
    print("Total amount of datapoints: " + str(len(raw_data)))
    print("Getting training data images")

    chunked_list = list(divide_chunks(raw_data, int((len(raw_data) / threadcount) + 1)))

    for i in range(threadcount):
        if(i > len(chunked_list) - 1):
            continue
        thread = Thread(target = get_single_data_instance, args = (img_size, chunked_list[i], data, mutex, counters))
        thread.start()
        threads.append(thread)

    print("Done starting threads")

    current_thread = 0
    for thread in threads:
        thread.join()
        current_thread += 1

    print("Ok: " + str(counters['ok']))
    print("Failed: " + str(counters['fail']))

    return data

def get_network(input_size, output_size):
    graph = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(input_size, input_size, 3)),
        tf.keras.layers.Dense(1000, activation='relu'),
        tf.keras.layers.Dense(500, activation='softmax'),
        tf.keras.layers.Dense(output_size)
    ])

    graph.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return graph

def train(input_network, training_input, training_output):
    return input_network.fit(training_input, training_output, 64, 10)

def make_prediction(input_network, image):
    predictions = input_network.predict(np.array([image]))
    guesses = np.argmax(predictions, axis=1)
    print(guesses)
    return guesses
