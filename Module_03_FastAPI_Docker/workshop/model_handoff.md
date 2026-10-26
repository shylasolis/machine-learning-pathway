# Model Handoff: What Would You Save After Training?

## A folder, not just a mysterious "weights file"

After training, you should be able to hand another person enough information
to reproduce **inference** without rerunning training. This workshop gives you
that handoff in [artifacts/](artifacts/).

Cloning or downloading the course repository downloads the folder. Alternatively,
download [model_handoff.zip](model_handoff.zip) from the course repository or
the copy supplied by your instructor. Extract it into this workshop folder:
the resulting `artifacts` folder must sit beside [app.py](app.py).
Do not end up with `artifacts/artifacts`. If your browser saves the ZIP in
Downloads, move it to the workshop folder before extracting it.

On Windows, right-click the ZIP -> **Extract All** and select the workshop
folder. On macOS, double-click the ZIP and move its `artifacts` folder beside
the API script if Finder creates an additional enclosing folder.

The ZIP is a model pack, not the complete application. You still need the
workshop source code and dependencies. Keep the supplied files together.
Open these JSON files in a text editor; do not edit their numbers for the lab.

| Supplied item | What it means | What you would save from your own trained model |
|---|---|---|
| `logistic_weights.json` | Two learned coefficients, an intercept, class ordering, and threshold; coefficients apply to standardized inputs | The model's `coef_`, `intercept_`, `classes_`, and documented decision rule |
| `preprocessing.json` | Training means/scales and required feature order | The fitted scaler's `mean_` and `scale_`; also fitted encoders/imputers if your model uses them |
| `teaching_pipeline.joblib` | Reloadable fitted scaler and logistic estimator together | A serialized fitted preprocessing/model pipeline |
| `model_metadata.json` | Model ID, source/context, feature contract, package versions, and reference output | A record of how to load/use the model and which training run it belongs to |
| `example_request.json` | A valid sample input | A documented input/output example for checking a handoff |

The plain JSON files make the learned values visible. The API loads the saved
**pipeline** to preserve scikit-learn's preprocessing and prediction behavior.
It does not load JSON weights into a second independently fitted model.
Changing a JSON number alone will not change that serialized pipeline. Rebuild
and verify the whole pack if intentionally changing the model.

**Security:** load joblib/pickle files only from a trusted source. A serialized
Python object can execute code during loading. A file extension or classmate's
claim is not a security guarantee. The API has no file-upload endpoint.

## Connect weights to the equation you know

The raw inputs are `mean_radius` and `mean_texture`. Before applying the
logistic coefficients, the saved scaler computes:

```text
x1_scaled = (mean_radius  - training_mean[0]) / training_scale[0]
x2_scaled = (mean_texture - training_mean[1]) / training_scale[1]
z = intercept + coefficient[0]*x1_scaled + coefficient[1]*x2_scaled
p = 1 / (1 + exp(-z))
```

The intercept is the constant term, sometimes called the bias. The two
coefficients are the feature weights. They are learned during training;
the API merely reuses them. A threshold is a **decision policy**, not an
additional learned feature coefficient. Our 0.50 threshold is fixed for this
demonstration, not presented as a clinically validated operating point.

## Inspect-and-explain activity

Before starting the API:

1. Open the weights JSON and identify the intercept and each feature's weight.
2. Open the preprocessing JSON and find the two training means and scales.
3. Explain why a raw radius of 14 must not be multiplied directly by a
   standardized-feature coefficient without first applying the saved scaler.
4. Open the metadata and find the class-1 meaning, threshold, build versions,
   and reference probability.
5. Identify which file(s) let a teammate run inference, and why weights alone
   would not communicate feature order, scaling, and class meaning reliably.

The unit test `test_readable_weights_and_scaler_reproduce_pipeline` independently
calculates probabilities from these readable values and checks them against
the saved pipeline. It is a consistency check, not a model-accuracy evaluation.

## If you trained your own model later

The essential pattern for a scikit-learn model is:

```python
pipeline.fit(X_train, y_train)
joblib.dump(pipeline, "teaching_pipeline.joblib")
```

Fit using training data only, finish your evaluation decisions, and save the
fitted pipeline without refitting it on API requests. Also save a feature schema,
class meanings, preprocessing details, library versions, chosen threshold,
source/training-run information, and a reference request/response.
See [prepare_model.py](prepare_model.py) for this pack's complete export process;
running it is **not required** for students receiving the supplied artifact.

A neural network's weights file additionally needs its architecture and matching
preprocessing/code. A tree model saves split structures rather than logistic
coefficients. "Save the weights" is not a universal complete deployment format.

Do not save private training records, passwords, API keys, or peer signatures
inside a model pack. This teaching artifact is deliberately small and has no
claim of clinical validity. Do not substitute it for Project 2's required fit.
