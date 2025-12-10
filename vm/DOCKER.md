# Docker Runner Guide

The AntClock repository includes a convenient Docker runner script (`docker_run.sh`) for running the Zero-image μVM in Docker containers.

## Prerequisites

- Docker installed and running
- Repository cloned locally
- Run commands from the repository root directory

## Quick Start

```bash
# Build the Docker image
./docker_run.sh build

# Run the VM (shows help by default)
./docker_run.sh run

# Run the VM with custom arguments
./docker_run.sh run --help
./docker_run.sh run --version
```

## Commands

### build

Builds the Docker image from `vm/Dockerfile`.

```bash
./docker_run.sh build
```

This creates an Alpine Linux-based image (~50 MB) containing:
- GCC compiler and build tools
- Compiled Zero-image μVM (~18 KB binary)
- /programs volume mount point

The image is tagged as `antclock-zero-vm:latest`.

### run

Runs the VM in a Docker container with the specified arguments.

```bash
# Show VM help
./docker_run.sh run --help

# Run a program from the mounted volume
./docker_run.sh run /programs/example.zero

# Run with custom VM options
./docker_run.sh run --verbose /programs/program.zero
```

**Volume Mounts:**
- `$(pwd)/vm/programs` → `/programs` (read/write)

Place your Zero-image program files in `vm/programs/` to access them from the container.

### shell

Opens an interactive shell inside the container for debugging and exploration.

```bash
./docker_run.sh shell
```

Inside the shell, you can:
- Navigate the filesystem
- Inspect the VM binary: `ls -lh /vm/zero_vm`
- Access mounted programs: `ls /programs`
- Run VM commands directly: `./zero_vm --help`
- Build and test programs: `cd /programs && ...`

Type `exit` to leave the container.

### clean

Removes the Docker image and cleans up dangling images.

```bash
./docker_run.sh clean
```

This frees up disk space by removing:
- The `antclock-zero-vm:latest` image
- Any dangling/unused Docker images

## Examples

### Example 1: Building and Running

```bash
# Build the image (first time only)
./docker_run.sh build

# Run with help
./docker_run.sh run --help
```

### Example 2: Creating and Running a Program

First, create a simple Zero-image program in `vm/programs/`:

```bash
# Create programs directory if it doesn't exist
mkdir -p vm/programs

# Build example programs from the vm/examples directory
cd vm/examples
make
cp *.zero ../programs/  # Copy built examples
cd ../..

# Run an example program
./docker_run.sh run /programs/simple_projection.zero
```

### Example 3: Interactive Debugging

```bash
# Open shell in container
./docker_run.sh shell

# Inside container:
# $ ls -la /vm
# $ ./zero_vm --help
# $ cd /programs
# $ ./zero_vm example.zero
# $ exit
```

### Example 4: Automated Workflows

The Docker runner automatically builds the image if it doesn't exist:

```bash
# This will build first, then run
./docker_run.sh run --help
```

### Example 5: Clean Rebuild

```bash
# Remove old image
./docker_run.sh clean

# Build fresh image
./docker_run.sh build

# Test
./docker_run.sh run --help
```

## Volume Mounting

The runner automatically mounts `vm/programs` to `/programs` in the container:

```
Host: /path/to/antclock/vm/programs/
  ↓
Container: /programs/
```

**Usage Tips:**
- Place `.zero` program files in `vm/programs/`
- Programs are accessible at `/programs/filename.zero` in container
- Changes made in container persist to host (read/write mount)
- Use relative paths from `/programs/` when running programs

## Troubleshooting

### Docker Not Found

```
✗ Docker is not installed or not in PATH
ℹ Please install Docker: https://docs.docker.com/get-docker/
```

**Solution:** Install Docker from https://docs.docker.com/get-docker/

### Docker Daemon Not Running

```
✗ Docker daemon is not running
ℹ Please start Docker daemon
```

**Solution:** Start the Docker daemon (varies by OS)

### Build Failures

If the Docker build fails (e.g., network issues with Alpine repositories):
1. Check your internet connection
2. Try again after a few minutes
3. Consider using a Docker mirror or proxy

### Image Not Found

The runner will automatically build the image if it doesn't exist:

```
⚠ Docker image not found
ℹ Building image first...
```

This is normal behavior on first run.

## Advanced Usage

### Custom Docker Commands

You can run Docker commands directly if needed:

```bash
# Build manually
cd vm && docker build -t antclock-zero-vm:latest .

# Run with custom settings
docker run --rm -it \
  -v $(pwd)/vm/programs:/programs \
  antclock-zero-vm:latest /programs/example.zero

# Inspect the image
docker images antclock-zero-vm

# Remove manually
docker rmi antclock-zero-vm:latest
```

### Using the Makefile

The `vm/Makefile` also has a docker target:

```bash
cd vm
make docker  # Builds the image as 'zero-vm:latest'
```

Note: The Makefile uses a different image name (`zero-vm`) than the runner script (`antclock-zero-vm`).

## Architecture

The Docker image:
- **Base:** Alpine Linux (latest) - minimal, secure
- **Size:** ~50 MB total (~18 KB VM binary + OS + build tools)
- **Build Time:** ~2-3 minutes (depending on network speed)
- **Runtime:** Ephemeral containers (--rm flag)

Each container:
- Runs the Zero-image μVM as entrypoint
- Mounts `vm/programs` for program access
- Cleans up automatically after execution
- Isolated from host system (except mounted volumes)

## See Also

- [vm/README.md](README.md) - Complete VM documentation
- [vm/Dockerfile](Dockerfile) - Image definition
- [vm/examples/](examples/) - Example programs
- [Main README.md](../README.md) - AntClock overview
