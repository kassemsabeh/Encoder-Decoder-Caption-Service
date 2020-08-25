# Attention Caption Service
This service is responsible for image captioning using an encoder-decoder network on image streams and the output is a vekg of each image.



# Commands Stream
## Inputs
...

## Outputs
...
# Data Stream
Expect an data event with at least the following fields, as the following example:
```json
{
    "id": "abc-123abc-123abc-123abc-123abc-123abc-123",
    "image_url": "c8d025d3-8c3a-460c-a6f5-cabb7b179807",
    "data_flow": [
        ["object-detection-data"],
        ["color-detection-data"],
        ["gb-data"]
    ],
    "data_path": [],
    "vekg": {},
}
```

Enrich the event data with the following fields, as the following example:
```json
 "vekg": {
    "nodes": [
        [
            "02fad514-6c93-4b09-969b-666e9cd52799",
             {
                 "id": "02fad514-6c93-4b09-969b-666e9cd52799",
                 "caption": "car moving down the street",
              }
        ],
    ]
  },
  "data_path": ["object-detection-data",],
}
```

# Installation

## Configure .env
Copy the `example.env` file to `.env`, and inside it replace `SIT_PYPI_USER` and `SIT_PYPI_PASS` with the correct information.

## Installing Dependencies

### Using pipenv
Run `$ pipenv shell` to create a python virtualenv and load the .env into the environment variables in the shell.

Then run: `$ pipenv install` to install all packages, or `$ pipenv install -d` to also install the packages that help during development, eg: ipython.
This runs the installation using **pip** under the hood, but also handle the cross dependency issues between packages and checks the packages MD5s for security mesure.


### Using pip
To install using pip directly, one needs to use the `--extra-index-url` when running the `pip install` command, in order for to be able to use our private Pypi repository.

Load the environment variables from `.env` file using `source load_env.sh`.

To install from the `requirements.txt` file, run the following command:
```
$ pip install --extra-index-url https://${SIT_PYPI_USER}:${SIT_PYPI_PASS}@sit-pypi.herokuapp.com/simple -r requirements.txt
```

# Running
Enter project python environment (virtualenv or conda environment)

**ps**: It's required to have the .env variables loaded into the shell so that the project can run properly. An easy way of doing this is using `pipenv shell` to start the python environment with the `.env` file loaded or using the `source load_env.sh` command inside your preferable python environment (eg: conda).

Then, run the service with:
```
$ ./object_detection_service/run.py
```

# Testing
Run the script `run_tests.sh`, it will run all tests defined in the **tests** directory.

Also, there's a python script at `./object_detection_service/send_msgs_test.py` to do some simple manual testing, by sending msgs to the service stream key.


# Docker
## Build
Build the docker image using: `docker-compose build`

**ps**: It's required to have the .env variables loaded into the shell so that the container can build properly. An easy way of doing this is using `pipenv shell` to start the python environment with the `.env` file loaded or using the `source load_env.sh` command inside your preferable python environment (eg: conda).

## Run
Use `docker-compose run --rm service` to run the docker image


## Gitlab CI auto-build and tests

This is automatically enabled for this project (using the `.gitlab-ci.yml` present in this project root folder).

By default it will build the Dockerfile with every commit sent to the origin repository and tag it as 'dev'.

Afterwards, it will use this newly builty image to run the tests using the `./run_tests.sh` script.

But in order to make the automatic docker image build work, you'll need to set the `SIT_PYPI_USER` and `SIT_PYPI_PASS` variables in the Gitlab CI setting page: [Object Detection Service CI Setting Page](https://gitlab.insight-centre.org/sit/mps/object-detection-service/settings/ci_cd).

And, in order to make the automatic tests work, you should also set the rest of the environement variables required by your service, in the this projects `.gitlab-ci.yml` file, in the `variables` section. But don't add sensitive information to this file, such as passwords, this should be set through the Gitlab CI settings page, just like the `SIT_PYPI_USER`.

## Benchmark Tests
To run the benchmark tests one needs to manually start the Benchmark stage in the CI pipeline, it shoud be enabled after the tests stage is done. Only by passing the benchmark tests shoud the image be tagged with 'latest', to show that it is a stable docker image.
