# Application entry point

from flask import Flask
from connexion import App
from classifier import api

app = App(__name__, specification_dir='./')

# Read the Swagger specification
app.add_api('swagger.yml')

# Entrypoint
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
