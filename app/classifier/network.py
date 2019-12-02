import tensorflow as tf
import numpy as np
import requests
from PIL import Image
from io import BytesIO
from SPARQLWrapper import SPARQLWrapper, JSON
from threading import Thread, Lock, Semaphore
import json
import config

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


def get_training_data_thread(num, offset, mutex, resdict):
    print("Thread started (offset %d)" % offset)
    locked_mutex = False
    try:
        sparql = SPARQLWrapper("https://api.data.netwerkdigitaalerfgoed.nl/datasets/hackalod/RM-PublicDomainImages/services/RM-PublicDomainImages/sparql")
        sparql.setQuery("""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX dc: <http://purl.org/dc/elements/1.1/>
        PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

        SELECT distinct ?img ?ic ?broader WHERE {
        ?src <http://www.europeana.eu/schemas/edm/isShownBy> ?img .
        ?src <http://www.europeana.eu/schemas/edm/isShownAt> ?sub .

        ?sub dc:subject ?ic .
        ?ic skos:broader ?broader .
        FILTER(CONTAINS(str(?ic), "http://iconclass.org")) .
        } LIMIT %d OFFSET %d
        """ % (num, offset))

        sparql.setReturnFormat(JSON)
        results =  sparql.query().convert()

        mutex.acquire()
        locked_mutex = True
        for result in results["results"]["bindings"]:
            if (result["img"]["value"] not in resdict.keys()):
                resdict[result["img"]["value"]] = []

            currentList = result["img"]["value"]
            if (result["ic"]["value"] not in currentList):
                resdict[result["img"]["value"]].append(result["ic"]["value"])

            if (result["broader"]["value"] not in currentList):
                resdict[result["img"]["value"]].append(result["broader"]["value"])
        mutex.release()
        locked_mutex = False
        print("Offset %d done" % offset)
    except Exception as e:
        print(e)
        if(locked_mutex):
            mutex.unlock()
        print("Error at offset %d, skipping" % offset)

def get_training_data_master(num, offsets, mutex, resdict):
    for offset in offsets:
        get_training_data_thread(num, offset, mutex, resdict)

def get_training_data_raw(k=10000, n=64):
    resdict = {}
    threads = list()
    mutex = Lock()
    num_threads = 64
    offsets = [k * i for i in range(n)]

    offsets = list(divide_chunks(offsets, num_threads))

    for offset in offsets:
        thread = Thread(target = get_training_data_master, args = (k, offset, mutex, resdict))
        thread.start()
        threads.append(thread)

    threadcounter = 0
    for threadobject in threads:
        print("Waiting for %d" % threadcounter)
        threadobject.join()
        threadcounter += 1

    return resdict

def get_single_data_instance(img_size, datapoints, data, lock, counters):
    localentries = list()
    localfailed = 0
    localok = 0

    for datapoint in datapoints:
        # Get image URL
        img = datapoint['img']
        img = img[:-3]
        img = img + '=s' + str(img_size)

        # Get actual image
        response = None
        tries = 0
        while tries < 4:
            try:
                response = requests.get(img, timeout=20)
                print("Succesfully retrieved entry (" + str(tries + 1) + " tries) with URL: " + str(img))
                localok += 1
                break
            except requests.RequestException:
                tries += 1

        if(tries >= 4):
            localfailed += 1
            print("Skipping entry with URL: " + img)
            continue

        img = Image.open(BytesIO(response.content))
        img = resize(img, img_size)
        img = np.array(img)

        ic = datapoint['ic']
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
    rawer_data = get_training_data_raw()
    raw_data = list()

    for imgkey in rawer_data:
        raw_data.append({'img': imgkey, 'ic': rawer_data[imgkey]})

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
    opt = tf.keras.optimizers.Adam(learning_rate=0.1)
    graph = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(input_size, input_size, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(output_size)
    ])

    graph.compile(
        optimizer=opt,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return graph

def train(input_network, training_input, training_output):
    scaled_input = training_input / 255
    scaled_input -= np.min(scaled_input)
    scaled_input /= np.max(scaled_input)

    return input_network.fit(scaled_input, training_output, 32, 1)

def make_prediction(input_network, image):
    np_image = np.array(image) / 255
    np_image -= np.min(np_image)
    np_image /= np.max(np_image)

    predictions = input_network.predict(np.array([np_image]))[0]
    predictions -= np.min(predictions)
    predictions /= np.max(predictions)

    to_return = []
    for i in range(len(predictions)):
        if predictions[i] > 0.75:
            to_return.append(config.labels[i])

    print(to_return)
    return json.dumps(to_return)
