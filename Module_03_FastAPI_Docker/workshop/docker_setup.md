# Before the Lab: Install Docker Desktop

**Audience:** first-time Docker users on Windows or macOS.  
**Allow:** 30-60 minutes before class; updates, restarts, and campus support may
take longer. Do not begin installation during the timed workshop.

## What is Docker, and is it free?

Imagine handing someone a recipe, ingredients, and the right kitchen setup
instead of only your finished meal. Docker packages an application's software
environment so it can run consistently on another compatible computer.
It does not magically fix code, replace a trained model, or make predictions
more accurate.

- **Docker Desktop:** the application you install on your laptop. It provides
  tools and a local engine for running containers.
- **Image:** a packaged application template; not a photo.
- **Container:** a running instance of an image.
- **Dockerfile:** the recipe used to build the image.
- **Docker Hub:** an online image registry. It is different from GitHub,
  which hosts our source repository.

**Yes: Docker Desktop is free for educational use**, according to Docker's
[license guidance](https://docs.docker.com/subscription-billing/desktop-license/).
Students do not need to buy a paid plan for this local class lab. This does not
mean every workplace qualifies for free use; commercial/government rules differ.
Cloud hosting and paid Docker features are not required.
Read the terms presented by your installer. Do not enter payment information
for this exercise. You do not need to publish an image to Docker Hub.
Some networks or pull limits may require a free account; a school-managed
installation may require its organization's sign-in.

## 1. Identify your computer

### Windows

1. Open **Start -> Settings -> System -> About**. Record the Windows edition,
   version, system type, and installed memory.
2. Most Windows laptops are x64/AMD64, including Intel processors. ARM devices
   need an ARM-compatible Docker installer and may have different support.
3. Open **Task Manager -> Performance -> CPU** and look for
   **Virtualization: Enabled**.
4. Compare your computer with Docker's current
   [Windows requirements](https://docs.docker.com/desktop/setup/install/windows-install/).
   The WSL 2 x64 path lists WSL 2.1.5 or later, 8 GB RAM, supported 64-bit
   Windows, and hardware virtualization. OS support changes; use that official
   page rather than assuming every Windows release is supported.

If virtualization is disabled or installation is blocked on a managed device,
ask campus IT or the instructor. Do not change firmware settings, disable
security software, or bypass school policies on your own.

### Mac

1. Open **Apple menu -> About This Mac**.
2. Record the macOS version and whether it says **Chip: Apple M...**
   (Apple silicon) or **Processor: Intel**.
3. Compare with Docker's current
   [Mac requirements](https://docs.docker.com/desktop/setup/install/mac-install/).
   Docker lists at least 4 GB RAM and supports the current and two previous
   major macOS releases. More available memory is useful for the lab.
4. Leave several GB of free disk space for Docker and downloaded images;
   the precise requirement depends on images and software updates.

## 2A. Install on Windows using WSL 2

WSL means **Windows Subsystem for Linux**. Docker uses it to run the Linux
environment needed by our container; you do not need to replace Windows.

1. Open **PowerShell** from Start. Run:

   ```powershell
   wsl --version
   wsl --status
   ```

2. If WSL is missing, follow Microsoft's
   [WSL installation instructions](https://learn.microsoft.com/windows/wsl/install).
   On your own authorized computer, open PowerShell **as Administrator**
   and run:

   ```powershell
   wsl --install
   ```

   Restart when instructed. If a Linux distribution asks you to create a
   username/password, complete that setup; its password entry may show no
   characters as you type. Keep the credentials private.

3. If WSL needs updating, follow the same official guidance and run:

   ```powershell
   wsl --update
   wsl --version
   ```

   Ask IT if permissions or Windows features block this. Do not repeatedly
   reinstall WSL or remove distributions containing your files.

4. Download **Docker Desktop for Windows** from the official Docker Windows
   installation page. Choose the installer matching your system architecture.
5. Run the installer and use the **WSL 2** backend when offered. Accept the
   license only after reading it. Follow any restart prompt.
6. Open **Docker Desktop** from Start. Complete onboarding without purchasing
   a subscription. Let the engine finish starting.
7. In Docker Desktop settings, verify **Use the WSL 2 based engine** if that
   option is available. Our lab needs **Linux containers**, not Windows
   containers. Do not change to Windows containers.
8. Close and reopen an ordinary PowerShell window, then continue to Step 3.
   You do not need to run everyday lab commands as Administrator.

## 2B. Install on Mac

1. Open Docker's official Mac installation page and choose **Apple silicon**
   or **Intel** based on Step 1. Do not choose by guessing.
2. Open the downloaded `Docker.dmg` and drag Docker into **Applications**.
3. Open **Applications -> Docker**. Read/accept its license and follow the
   installer prompts. Ask IT about any administrator restrictions.
4. Complete onboarding without purchasing a subscription. Wait for the engine
   to be running. Leave Docker Desktop open in the background.
5. Open **Terminal** using Command+Space -> type Terminal -> Return.
6. Continue to Step 3. Rosetta is not a prerequisite for our native
   multi-architecture Python image; if another Docker tool requests it, follow
   Docker's official guidance rather than installing unrelated software.

## 3. Prove that installation works (both systems)

In PowerShell or Terminal, run one line at a time:

```text
docker --version
docker version
docker run --rm hello-world
```

Expected results:

- `docker --version` prints the installed command-line tool's version.
- `docker version` prints **both Client and Server** information. Client
  information alone does not prove the engine is running.
- `docker run --rm hello-world` downloads a tiny demonstration image and prints
  **Hello from Docker!**. The demonstration container then exits and is removed.
  It does not start our model service.

The first pull requires internet access. Do not paste private account tokens
or your full system information into public Git issues.

### If a check fails

| Problem | Next step |
|---|---|
| `docker` is not recognized / command not found | Finish installation, reopen the terminal, and check Docker's official PATH/setup instructions. |
| Cannot connect to daemon/engine; named pipe or socket error | Open Docker Desktop and wait until the engine is running. On Windows confirm WSL readiness and Linux-container mode. |
| WSL or virtualization error | Capture the error and ask IT/instructor; do not bypass device-management policy. |
| `hello-world` cannot download, or a pull limit appears | Check internet/proxy settings and the Docker message; sign in with a free account if required. Do not buy a plan for the lab. |
| Unsupported OS or insufficient hardware | Tell the instructor before class. Arrange an approved supported lab computer; do not claim a completed Docker run. |

## 4. Bring your readiness evidence

Record your OS/chip, Docker client/server versions, and whether
`Hello from Docker!` appeared. An optional cropped screenshot may show only the
success message, not personal information.

You are ready when Docker Desktop runs and all three checks succeed.
Then return to the [guided workshop](README.md). Python/Git setup is also
required; Docker installation alone does not install the host Python tools.

The class goal includes building/running the model container, not just
installing the Docker command. If your setup is blocked, notify the instructor
and complete the native API activities while an approved solution is arranged.
