# Module 3: FastAPI & Docker Containerization

## Place in the course

The [syllabus diagram](../syllabus.pdf), page 3, distinguishes foundational
modules from applied projects. **Module 3 is FastAPI & Docker Containerization**;
clustering is **Project Assignment 3**, not Module 3. The topic table on page 7
names model serialization, reproducible inference preprocessing, FastAPI
services, and Docker packaging.

Module 1 established data procurement and EDA. Module 2 established preparation,
training, and evaluation. Project 1 applied those ideas to regression; Project 2
applies them to classification. Module 3 asks: **How can another program use a
trained model reliably without retraining it?** Module 4 will address production
MLOps, monitoring, and drift.

## Starting point

The instructor reported reaching slide 15 of the current logistic regression
lecture on October 19. Do not assume students have completed classification
training, threshold evaluation, or Project 2. The first workshop supplies a
trained model; students need only understand features, a probability, and a
threshold. Briefly review those concepts before introducing the service.

## Learning outcomes

Students will be able to:

1. Distinguish training from inference and explain why an API must not fit a
   scaler or model on incoming requests.
2. Identify a model artifact and its feature, preprocessing, version, and
   class-label contract.
3. Explain an HTTP request, JSON body, endpoint, and response.
4. Run a local FastAPI service and send valid and invalid requests.
5. Verify that API predictions match the supplied fitted pipeline.
6. Explain the differences between source code, an image, and a running
   container, including host/container port mapping.

## Module sequence

| Stage | Focus | Evidence of understanding |
|---|---|---|
| Pre-lab: computer readiness | Install Docker Desktop, Python, and Git; verify the Docker engine | Docker client/server output and successful hello-world container |
| First lesson: 90-minute guided workshop | Supplied weights/scaler/pipeline handoff, local API, validation, and container run | A successful request, rejected request, reference check, and equivalent container prediction |
| Follow-up practice | Reproduce the container on another computer; investigate setup failures | Peer explanation and documented environment/port behavior |
| Project integration | Package a student's already evaluated model with its fitted preprocessing | Professional README and independent replication; preserve the project's evaluation policy |
| Bridge to Module 4 | What this local demo does not provide: authentication, operational checks, monitoring, drift handling | Explain why a working API is not automatically a production system |

Docker setup and a successful model-container run are module goals, not optional
extras. Resolve installation before class where possible. A blocked student can
practice the native API while arranging an approved supported computer, but
must not report that as completed Docker validation.

## First-lesson schedule

Complete software installation before class; installation time is not included.

| Minutes | Activity |
|---|---|
| 0-10 | Recap score -> probability -> class; contrast training with inference |
| 10-25 | Inspect readable weights/scaler values and trace the model handoff |
| 25-40 | Start FastAPI; open `/docs`; distinguish server terminal from browser |
| 40-55 | Submit a prediction and interpret its response fields |
| 55-65 | Explore validation errors and run the reproducibility check |
| 65-85 | Build/run the Docker image and compare the container's response |
| 85-90 | Explain the request flow to a peer and complete the exit questions |

## Workshop

Start with the [beginner workshop instructions](workshop/README.md).
Begin with the [Docker setup pre-lab](workshop/docker_setup.md) and
[model handoff guide](workshop/model_handoff.md). The workshop includes
Windows/macOS setup, exact run commands, expected results, troubleshooting,
the container run, and an exit checkpoint. Pre-pull/build the image after setup
if campus download speeds would otherwise consume class time.

For a print-friendly walkthrough, use the combined
[Module 3 lab guide PDF](workshop/Module_03_FastAPI_Docker_Lab_Guide.pdf).

This is a **guided foundational workshop**, not a replacement for Project
Assignment 2 and not a newly assigned graded project. Project deliverables and
portfolio expectations remain in the
[Classification assignment](../Project_02_Classification/assignment_instructions.md).

## Teaching boundaries

- The supplied Wisconsin Diagnostic Breast Cancer example deliberately reuses
  the classroom classification context. It is not a clinical service, medical
  advice, a new real-world dataset selection, or a production deployment.
- Its regularized, training-only demonstration fit is separate from Project 2's
  prescribed unregularized comparison and validation-selected threshold.
  Do not copy these model settings into that experiment or claim new evaluation
  results for this workshop.
- Only synthetic example measurements are sent to the local endpoint; do not
  collect personal medical information.
- `joblib` artifacts can execute code when loaded. Use only the supplied trusted
  artifact, never an uploaded or unfamiliar pickle/joblib file.
- A health response means the service loaded its model, not that its predictions
  are accurate, calibrated, fair, or suitable for deployment.

## Verification status

See the workshop's verification section for executed local checks and any
platform-dependent limitations. The presence of Docker instructions alone
does not establish that a container has been built and tested.
