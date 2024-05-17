# Assignment 7

The `app` directory contains `task.py` and `task_docker.py`. You can run `uvicorn app.task:app -p 8000` to run the FastAPI server. Then, at `localhost:8000/docs` 
the Swagger UI can be opened and used for the `/predict` endpoint.

The prometheus server can be run by installing the executable and running `./prometheus`. Note that the `prometheus.yaml` file should be in the same directory as the executable,
and it determines some of the configuration settings for the executable.

At `localhost:9090` we can access the prometheus server, where we can observe the response metrics that are returned by the API endpoint at the path `/metrics`.

FInally, grafana can be set up using brew and linked to `localhost:9090` in order to observe interesting plots of the metrics.

<img width="723" alt="image" src="https://github.com/nikhilanand03/prometheus-docker-bdl-a07/assets/75153414/2cbfbc36-0770-4690-b58d-206cc554cd70">
