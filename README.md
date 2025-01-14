## Prerequisites

In order to start the project, first you need a couple of things:
Create a virtual environment

```
python -m venv <venv-name>
```

Activate the virtual environment. For linux execute the following command:

```
source <venv-name>/bin/activate
```

and for windows:

```
source <venv-name>/Scripts/activate
```

Then you need to install tha package into the virtual environment.
The example below will make an editable install of the package (more can be read here: https://setuptools.pypa.io/en/latest/userguide/development_mode.html). Take note the command has to be executed within an activated virtual environment and from the root folder, where the pyproject.toml file is located.

For this current setup it is under /code.

```
pip install -e .
```

The installation make take a couple of minutes depending on the machine on which it is installed.

After that, there are two ways to use the project.

## Running the API

To run the api we need the uvicorn server.
There are two options to start the API. First one is via:

```
uvicorn cm_evaluator.api.main:app --reload
```

You can also specify various options like port, reload, etc (more on options can be read here: https://www.uvicorn.org/settings/)

After that you can navigate to "<host>:<port>/docs" to use the swagger docs for the API (default under localhost will be http://127.0.0.1:8000/docs#)
