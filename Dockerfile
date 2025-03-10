FROM docker.io/python:3.11.6

WORKDIR /workspace

# Use PDM to keep track of exact versions of dependencies
RUN pip install pdm
COPY pyproject.toml README.md pdm.lock ./
# install dependencies first. PDM also creates a /workspace/.venv here.
ENV PATH="/workspace/.venv/bin:$PATH"
RUN pdm install  --no-self
COPY examples ./examples
COPY robot_search ./robot_search
COPY funsearch ./funsearch

RUN pip install --no-deps . 
RUN llm install llm-ollama

RUN pip install mujoco==3.2.4 
RUN pip install dm_control==1.0.24

# if running the container
RUN rm -r ./funsearch ./build
CMD /bin/bash

# if debugging
#RUN pip install debugpy
#CMD ["python", "-Xfrozen_modules=off", "-m", "debugpy", "--listen", "0.0.0.0:5678", "--wait-for-client", "funsearch", "run", "examples/inv_pendulum_spec.py", "0.6", "--sandbox_type", "ExternalProcessSandbox"]