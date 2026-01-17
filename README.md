# MSBA265
Special Analytics Topics

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



