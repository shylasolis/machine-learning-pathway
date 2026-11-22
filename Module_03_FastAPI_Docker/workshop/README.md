# Guided Workshop: From a Trained Logistic Model to a Local API

**Duration:** 90 minutes after software setup.  
**Goal:** use a supplied trained model, not train a new one.  
**Safety:** classroom example only; not a medical diagnostic application.

Prefer one printable reference? Download the complete
[Module 3 lab guide PDF](Module_03_FastAPI_Docker_Lab_Guide.pdf). It combines
the Docker setup pre-lab, model handoff explanation, and this guided workshop.

Before class, complete the [Docker Desktop setup guide](docker_setup.md),
including the client/server and hello-world checks. Docker Desktop is free for
educational use under its current terms; no paid cloud hosting is used here.
Review the [model handoff guide](model_handoff.md) to learn what the supplied
weights, preprocessing values, and fitted pipeline represent.

## 1. Before class: get the software and project

You need internet access for installation, approximately 1 GB of free space
for Python/packages, and more space if using Docker. This workshop uses
**standard Python 3.14**, matching the supplied artifact's environment; do not
choose the experimental free-threaded build. An editor is optional.

- Install Python from [python.org](https://www.python.org/downloads/). On Windows,
  enable the Python command/PATH option if the installer offers it. On macOS,
  complete the installer, including its certificate setup if prompted.
- Install Git from [git-scm.com](https://git-scm.com/downloads).
- Close and reopen your terminal after installing software.
- Docker setup is a required pre-lab. Follow the separate step-by-step guide
  for your OS and processor; start Docker Desktop and verify that its engine
  can run a container before class.

**Open a terminal:** on Windows, search Start for **PowerShell**. On a Mac,
press Command+Space, type **Terminal**, and press Return. A terminal accepts
typed commands; press Enter/Return after each line. Do not type the Markdown
backticks surrounding examples.

If you already have this course repository, open its root folder in your
terminal and skip cloning. Otherwise, the following commands create a folder
named `MSBA265` in your terminal's current directory. If the instructor provides
a workshop branch, switch to that branch before continuing.

### Windows PowerShell

```powershell
git --version
py -3.14 --version
git clone https://github.com/shylasolis/MSBA265.git
cd .\MSBA265\Module_03_FastAPI_Docker\workshop
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

These Windows commands call the environment's Python directly, so activation
and PowerShell execution-policy changes are unnecessary.

### macOS Terminal

```bash
git --version
python3.14 --version
git clone https://github.com/shylasolis/MSBA265.git
cd MSBA265/Module_03_FastAPI_Docker/workshop
python3.14 -m venv .venv
./.venv/bin/python -m pip install -r requirements-dev.txt
```

If starting from the existing course repository root, navigate only into
`Module_03_FastAPI_Docker` and then `workshop` using your OS's path convention.
All remaining commands assume that **workshop is your current folder**.
Installation may take several minutes. Continue only if it finishes without
errors. Pinned primary package versions are in
[requirements.txt](requirements.txt); the development file adds the API test client.

## 2. Understand the supplied model (minutes 0-25)

Training adjusts model coefficients using labeled examples. **Inference** uses
the already learned coefficients on new measurements. An **artifact** is a
saved training result that another process can load.

Open [model_metadata.json](artifacts/model_metadata.json) in a text editor.
It documents the source, feature order, threshold, package versions, and a
reference request. The paired
[teaching_pipeline.joblib](artifacts/teaching_pipeline.joblib) contains:

```text
raw measurements -> fitted StandardScaler -> fitted LogisticRegression
```

Do not open the binary joblib file as text. Do not replace it with an artifact
from an unfamiliar source: joblib/pickle can execute code during loading.
The service does not accept uploaded model files.

The two features, in order, are `mean radius` and `mean texture` from
scikit-learn's bundled Wisconsin Diagnostic Breast Cancer dataset. Radius is
a cell-nucleus size measurement and texture describes grayscale variation;
these are not a person's age or body temperature. The API field names use
underscores: `mean_radius`, then `mean_texture`.

The provided model defines **1 = malignant** and **0 = benign**. This differs
from the bundled dataset's original coding. The probability is for class 1.
The decision rule is `p >= 0.50` -> class 1; otherwise class 0.

The training-only scaler and coefficients are already fitted. **Never fit a
scaler on a single API request.** The saved pipeline transforms new raw features
using the training means and scales. Sending already standardized inputs would
scale them twice and produce incorrect predictions.

[prepare_model.py](prepare_model.py) documents how the instructor built the
artifact. You do **not** need to run it, finish Project 2, or download any data.
This demo uses a regularized model and fixed threshold; it is not Project 2's
training or evaluation experiment.

**Discuss:** which files contain learned values, which contain instructions,
and why must the feature order remain consistent?

## 3. Start the service (minutes 25-40)

An **API** is an interface that lets programs exchange requests and responses.
FastAPI defines the interface; Uvicorn is the server process that listens for
requests.

Think of the browser as a customer, a JSON request as an order, the API as the
agreed ordering format, and the saved model as the calculation behind the
counter. The server waits for orders; it does not retrain the model for each one.

| First-time API vocabulary | Meaning in this lab |
|---|---|
| Client | Your browser, or the Python checker sending a request |
| Server | The running Uvicorn process hosting the FastAPI application |
| Endpoint | A named API route such as `/health` or `/predict` |
| Request body | The JSON input measurements sent with POST |
| Response | JSON returned by the service |
| Status code | HTTP result indicator: 200 for success, 422 for invalid input |
| Input schema | Required names/types/limits described by `PredictionRequest` |
| `/docs` | FastAPI's generated interactive interface for trying requests |

Unlike the Matplotlib GUI, this API has no sliders or prediction chart.
`/docs` supplies a convenient test screen; another program can use the same
endpoint without that screen. Opening a Python file is not the same as starting
its server.

In your workshop terminal, run the command for your OS:

Windows:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```

macOS:

```bash
./.venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

`app:app` means the `app` object inside [app.py](app.py).
`127.0.0.1` is your own computer; `8000` is the port, a numbered connection
point. This local lesson does not expose the service to the public internet.

Wait for **Application startup complete**. The terminal remains busy because
the server is running; this is expected. Leave it open.

In a web browser, open:

- `http://127.0.0.1:8000/health` to see a ready status and model ID.
- `http://127.0.0.1:8000/docs` to use the interactive API documentation.

The documentation screen may need internet access to load its interface assets;
the prediction endpoint itself does not use an external prediction service.

## 4. Send a prediction (minutes 40-55)

In `/docs`, expand **POST /predict**, click **Try it out**, replace the example
request with the following JSON, then click **Execute**:

```json
{
  "mean_radius": 14.0,
  "mean_texture": 20.0
}
```

JSON uses double-quoted field names, colons between keys and values, and commas
between fields. Numbers are not quoted. **POST** sends a body to the endpoint;
**GET** retrieves a resource, such as health information.

The response should have HTTP status **200** and approximately:

```json
{
  "model_id": "logistic-teaching-v1",
  "probability_class_1": 0.3265635181,
  "predicted_class": 0,
  "label": "benign",
  "threshold": 0.5,
  "positive_class": "malignant",
  "educational_only": true
}
```

Tiny floating-point differences are acceptable. This is a model output for a
classroom example, not a diagnosis. A successful response does not establish
accuracy or clinical validity.

Trace the request in [app.py](app.py): schema validation -> raw feature array
in fixed order -> saved pipeline -> class-1 probability -> threshold -> response.
The model loads once at startup, not once per request, and there is no training
endpoint.

## 5. Reject invalid inputs and verify results (minutes 55-65)

Try each of these requests separately in `/docs`:

1. Remove `mean_texture`.
2. Set `mean_radius` to `-1`.
3. Send `"14"` as text instead of `14` as a number.
4. Add an extra field such as `"age": 50`.

Each should return **422**, with details about which input failed validation.
Restore the valid request and confirm it returns 200 again. These errors are
part of the interface contract, not evidence that the model retrained.
The demonstration accepts finite numeric values greater than 0 and no more
than 100 for either feature. These are simple classroom guardrails, not
validated biological limits or a check for realistic/in-distribution data.

Open a **second terminal**, navigate to the same workshop folder, and keep the
server running in the first terminal. Execute:

Windows:

```powershell
.\.venv\Scripts\python.exe check_api.py
.\.venv\Scripts\python.exe -m unittest discover -s . -p "test_*.py" -v
```

macOS:

```bash
./.venv/bin/python check_api.py
./.venv/bin/python -m unittest discover -s . -p "test_*.py" -v
```

`check_api.py` sends real HTTP requests to the server and compares the prediction
to the supplied metadata. Expect `PASS: health and prediction match the supplied
reference.` The unit tests use an in-process client and do not need the server;
they check the saved pipeline, invalid requests, threshold equality, and that
inference does not refit or change the scaler.

The server normally starts in seconds once dependencies are installed. Training
is not part of request latency. This workshop prints responses; it does not
create Project 2's evaluation CSVs or overwrite its results.

## 6. Package and run the service in Docker (minutes 65-85)

An **image** packages the application and its dependencies. A **container** is a
running instance of that image. [Dockerfile](Dockerfile) describes how to build
the image; [.dockerignore](.dockerignore) prevents local environments and caches
from entering the build context. The trained artifact is copied, not retrained.

Complete this activity after the Docker readiness pre-lab. These commands are
the same on both operating systems. If setup remains blocked, practice the
native API and arrange an approved supported computer with the instructor;
the module's Docker checkpoint remains incomplete until you run it.
First stop the native server with **Ctrl+C** in its terminal.
From the workshop folder:

```text
docker version
docker build -t msba265-logistic-api .
docker run --rm --name msba265-logistic-workshop -p 127.0.0.1:8000:8000 msba265-logistic-api
```

The first build downloads a Python base image and packages and can take several
minutes. Inside the container, Uvicorn binds to `0.0.0.0` so the port can be
reached through Docker. The host mapping binds only to `127.0.0.1`, keeping the
lesson local. `EXPOSE 8000` documents a port; it does not publish it by itself.
The process runs as an unprivileged user.

Repeat `/health`, the `/docs` prediction, and `check_api.py` from the host's
second terminal. The reference prediction should not change.
Stop the container with Ctrl+C, or in another terminal use:

```text
docker stop msba265-logistic-workshop
```

`--rm` removes this stopped container, not your source files or the image.
Do not run broad container/image cleanup commands on a shared machine.

## 7. Exit checkpoint (minutes 85-90)

Keep brief notes for class discussion; this is not a new graded project.

1. Explain training versus inference and identify what is saved.
2. Explain why the scaler must travel with the model and why feature order matters.
3. Show a valid response and a 422 response; explain the difference.
4. Describe the full request path without saying that the API "trains a prediction."
5. Explain image versus container, and what `127.0.0.1:8000:8000` maps.
6. Name two reasons this local service is not production-ready.

Also retain the readiness check and evidence that the **container**, not just
the native Python process, returned the reference prediction. If the Docker
build is slow, finish it in follow-up practice rather than claiming it ran.

For portfolio integration later, retain professional Git organization,
beginner Windows/macOS instructions, and independent peer validation. Any
completed signed course validation form belongs on Canvas, not in Git.

## Troubleshooting

| Symptom | What to check |
|---|---|
| `git`, `py`, or `python3.14` not found | Install the named software, reopen the terminal, and verify its version. On Windows, if the launcher is unavailable but `python --version` shows 3.14, use `python -m venv .venv` for creation. |
| No `requirements-dev.txt` or cannot import `app` | You must be inside `Module_03_FastAPI_Docker/workshop`, not the repository root. Check the folder name in your terminal prompt. |
| `ModuleNotFoundError` | Install dependencies with the same `.venv` Python used in the run command. Do not use a different system Python to launch Uvicorn. |
| Missing artifact or version mismatch | Restore the supplied artifact/metadata pair and install the pinned requirements. Do not bypass the version check or substitute an unknown joblib file. |
| Address already in use | Stop your previous workshop server/container. Do not terminate unrelated processes. If another user owns port 8000, ask the instructor for a different port and update the URLs, checker, and Docker host mapping together. |
| Browser connection refused or checker fails to connect | Keep the server terminal open; read its startup error and confirm the host/port. If startup failed, correct the error before proceeding. |
| `/` gives 404 or browser `/predict` gives 405 | There is no home page; use `/docs`. Typing `/predict` in the address bar sends GET, but prediction requires POST. |
| Docker cannot connect to its engine | Start Docker Desktop, wait for readiness, and check Linux-container mode. Continue the native workshop while installation is resolved. |
| Docker build fails on a package or base-image download | Check internet access and Docker's connection. Capture the error; do not claim a successful container run. |

## Verification

The supplied model was built with Python 3.14.3 and the primary versions pinned
in [requirements.txt](requirements.txt). The saved model records its build
versions in the metadata. Pinning these libraries improves reproducibility;
it is not a complete cross-platform lockfile or a guarantee of clinical validity.

Executed on the local Windows Python 3.14.3 environment:

- All six unittest cases passed, including readable-weights/scaler equivalence,
  invalid input rejection, threshold equality, and no refitting during requests.
- A real Uvicorn server returned ready health and a prediction matching the
  supplied reference; the temporary test server was then stopped.
- Docker is not installed/on `PATH` in the environment used for this check.
  The image build/container run therefore remains **unverified**.
- macOS setup and execution have not been run here.

Do not claim a platform or container was tested based solely on instructions.
