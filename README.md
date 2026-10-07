# MSBA265
Special Analytics Topics

## Course structure

Foundational **modules** teach reusable methods. Applied **projects** use those
methods to solve a business problem; they are not additional module numbers.
The naming below matches the roadmap and milestone table in [the syllabus](syllabus.pdf).

| Type | Topic | Repository location |
|---|---|---|
| Module 1 | Data procurement, EDA, and outliers | [Module_01_EDA](Module_01_EDA/) |
| Module 2 | Preprocessing, models, and diagnostics | [Module_02_Preprocessing_Models_Diagnostics](Module_02_Preprocessing_Models_Diagnostics/) |
| Project 1 | Linear regression | [Project_01_Linear_Regression](Project_01_Linear_Regression/) |
| Project 2 | Classification | [Project_02_Classification](Project_02_Classification/) |
| Module 3 | FastAPI and Docker containerization | Planned |
| Module 4 | Production MLOps, monitoring, and drift | Planned |
| Project 3 | Clustering | Planned |
| Project 4 | NLP and sentiment | Planned |
| Capstone | End-to-end business analytics system | Planned |

Modules are introduced and reinforced as projects need them; this table is not
a dated teaching schedule. Current progress: Module 1 and introductory Module 2
content have supported Project 1. Project 2's
[classification foundation lecture](Project_02_Classification/Classification_Foundations_Lecture.pptx)
and [logistic regression masterclass](Project_02_Classification/Logistic_Regression_Masterclass.pptx)
are ready; its image dataset, guided lab, and assignment are still being selected.

The former `Module_02_Linear_Regression` folder has been separated into Module 2
foundations and Project 1 application materials. The combined Module 1/2 study
guide and training-strategies handout are in Module 2; regression scripts,
the lab notebook, lecture, and assignment are in Project 1.
Instructor-only materials retain `_Instructor` directories and remain excluded
by the existing Git ignore policy.

# MSBA 265 – JupyterLab Setup (Windows + Anaconda)

These steps avoid common NumPy/Pandas version conflicts by **not using the base environment**.

---

## 0) Install / Open Anaconda Prompt
- Install **Anaconda** (or Miniconda)
- Open **Anaconda Prompt** (from the Start menu)

You should see something like:

(base) C:\Users\YOURNAME>

---

## 1) Create a dedicated course environment
In Anaconda Prompt, run:

```bat
conda create -n msba265 python=3.11 -y
```

2) Activate the environment
```bat
conda activate msba265
```

Your prompt should now look like:
```scss
(msba265) C:\Users\YOURNAME>
```

3) Install course packages
```bat
conda install -c conda-forge numpy pandas jupyterlab -y
```

4) Launch JupyterLab
```bat
jupyter lab
```

5) Verify installation

In a new notebook, run:
```python
import numpy as np
import pandas as pd

print("numpy:", np.__version__)
print("pandas:", pd.__version__)
```

Common Issues
Jupyter opens but imports fail

Make sure JupyterLab was launched after activating the environment:

```bat
conda activate msba265
jupyter lab
```

Wrong kernel in Jupyter

In the notebook:
Kernel → Change Kernel → Python [conda env: msba265]

Important (All Students)

⚠️ Do NOT use the base environment for this course.
Always activate msba265 before launching JupyterLab.

Do NOT use the base conda environment for this course.
Always activate msba265 before launching JupyterLab.



# MSBA 265 – JupyterLab Setup (macOS + Anaconda)

These steps are the same as Windows, but use Terminal instead of Anaconda Prompt.

```scss
(base) username@MacBook ~ %
```

## 1) Create a dedicated course environment

In Terminal, run:
```bash
conda create -n msba265 python=3.11 -y
```

## 2) Activate the environment
```bash
conda activate msba265
```

Your prompt should now look similar to:
```scss
(msba265) username@MacBook ~ %
```

## 3) Install course packages
```bash
conda install -c conda-forge numpy pandas jupyterlab -y
```

## 4) Launch JupyterLab
```bash
jupyter lab
```

A browser window should open automatically.

## 5) Verify your setup

Create a new notebook and run:
```python
import numpy as np
import pandas as pd

print("numpy:", np.__version__)
print("pandas:", pd.__version__)
```
If this runs without errors, your setup is correct.

Important (All Students)

⚠️ Do NOT use the base environment for this course.
Always activate msba265 before launching JupyterLab.

Do NOT use the base conda environment for this course.
Always activate msba265 before launching JupyterLab.
