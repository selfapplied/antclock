#!/bin/bash
# Docker Runner for AntClock Zero-image μVM
# Builds and launches the VM in a Docker container
# Usage:
#   ./docker_run.sh build              # Build Docker image
#   ./docker_run.sh run [args...]      # Run VM with arguments
#   ./docker_run.sh shell              # Open shell in container
#   ./docker_run.sh clean              # Remove Docker image
#   ./docker_run.sh help               # Show help

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Docker configuration
IMAGE_NAME="antclock-zero-vm"
IMAGE_TAG="latest"
FULL_IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"
CONTAINER_NAME="antclock-vm-instance"

# Print functions
print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_header() {
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

# Check if Docker is available
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed or not in PATH"
        print_info "Please install Docker: https://docs.docker.com/get-docker/"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        print_error "Docker daemon is not running"
        print_info "Please start Docker daemon"
        exit 1
    fi
    
    print_status "Docker is available ($(docker --version))"
}

# Build Docker image
build_image() {
    print_header "Building AntClock Zero-image μVM Docker Image"
    
    check_docker
    
    if [ ! -f "vm/Dockerfile" ]; then
        print_error "vm/Dockerfile not found"
        print_info "Make sure you're running this script from the repository root"
        exit 1
    fi
    
    print_info "Building Docker image: ${FULL_IMAGE}"
    print_info "Build context: vm/"
    
    cd vm
    docker build -t "${FULL_IMAGE}" .
    cd ..
    
    print_status "Docker image built successfully: ${FULL_IMAGE}"
    
    # Show image info
    print_info "Image details:"
    docker images "${IMAGE_NAME}" --format "  Size: {{.Size}}, Created: {{.CreatedSince}}"
}

# Check if image exists
image_exists() {
    docker images -q "${FULL_IMAGE}" 2>/dev/null | grep -q .
}

# Ensure image is built
ensure_image() {
    if ! image_exists; then
        print_warning "Docker image not found"
        print_info "Building image first..."
        build_image
    fi
}

# Run VM in container
run_vm() {
    print_header "Running AntClock Zero-image μVM in Docker"
    
    check_docker
    ensure_image
    
    # Create programs directory if it doesn't exist
    mkdir -p "$(pwd)/vm/programs"
    
    print_info "Starting Docker container..."
    print_info "Volume mount: $(pwd)/vm/programs -> /programs"
    
    if [ $# -eq 0 ]; then
        # Run with default --help
        print_info "Running: zero_vm --help"
        docker run --rm \
            --name "${CONTAINER_NAME}" \
            -v "$(pwd)/vm/programs:/programs" \
            "${FULL_IMAGE}"
    else
        # Run with provided arguments
        print_info "Running: zero_vm $*"
        docker run --rm \
            --name "${CONTAINER_NAME}" \
            -v "$(pwd)/vm/programs:/programs" \
            "${FULL_IMAGE}" "$@"
    fi
    
    print_status "Container execution completed"
}

# Open interactive shell in container
run_shell() {
    print_header "Opening Interactive Shell in Container"
    
    check_docker
    ensure_image
    
    print_info "Starting interactive container..."
    print_info "Volume mount: $(pwd)/vm/programs -> /programs"
    print_info "Type 'exit' to leave the container"
    
    docker run --rm -it \
        --name "${CONTAINER_NAME}" \
        -v "$(pwd)/vm/programs:/programs" \
        --entrypoint /bin/sh \
        "${FULL_IMAGE}"
    
    print_status "Shell session ended"
}

# Clean up Docker image
clean_image() {
    print_header "Cleaning Up Docker Image"
    
    check_docker
    
    if image_exists; then
        print_info "Removing Docker image: ${FULL_IMAGE}"
        docker rmi "${FULL_IMAGE}"
        print_status "Docker image removed"
    else
        print_warning "Docker image ${FULL_IMAGE} not found"
    fi
    
    # Clean up dangling images
    if docker images -f "dangling=true" -q | grep -q .; then
        print_info "Removing dangling images..."
        docker image prune -f
        print_status "Dangling images removed"
    fi
}

# Show help
show_help() {
    cat << EOF
AntClock Docker Runner - Zero-image μVM Container Management

SYNOPSIS
    ./docker_run.sh COMMAND [OPTIONS]

DESCRIPTION
    Manages Docker containers for the AntClock Zero-image μVM.
    Provides convenient commands to build, run, and manage the VM in Docker.

COMMANDS
    build               Build the Docker image from vm/Dockerfile
                        Creates image: ${FULL_IMAGE}

    run [ARGS...]       Run the VM in a Docker container
                        Arguments are passed directly to zero_vm
                        Examples:
                          ./docker_run.sh run --help
                          ./docker_run.sh run /programs/example.zero
                          ./docker_run.sh run --version

    shell               Open an interactive shell in the container
                        Useful for debugging and exploration
                        Volume /programs is mounted for program access

    clean               Remove the Docker image and clean up
                        Removes ${FULL_IMAGE} and dangling images

    help                Show this help message

VOLUME MOUNTS
    The following directory is mounted into the container:
    - $(pwd)/vm/programs -> /programs (read/write)
    
    Place your Zero-image programs in vm/programs/ to access them
    from within the container at /programs/

EXAMPLES
    # Build the Docker image
    ./docker_run.sh build

    # Run VM with default help
    ./docker_run.sh run

    # Run VM with a program
    ./docker_run.sh run /programs/example.zero

    # Open interactive shell
    ./docker_run.sh shell

    # Clean up Docker image
    ./docker_run.sh clean

DOCKER IMAGE
    Name:        ${IMAGE_NAME}
    Tag:         ${IMAGE_TAG}
    Full name:   ${FULL_IMAGE}
    Base:        alpine:latest
    Size:        ~50 MB
    VM binary:   ~18 KB

REQUIREMENTS
    - Docker installed and running
    - Repository cloned locally
    - Run from repository root directory

SEE ALSO
    vm/README.md        - Zero-image μVM documentation
    vm/Dockerfile       - Docker image definition
    vm/examples/        - Example VM programs

EOF
}

# Main execution
main() {
    if [ $# -eq 0 ]; then
        print_error "No command provided"
        echo ""
        show_help
        exit 1
    fi

    COMMAND="$1"
    shift

    case "$COMMAND" in
        build)
            build_image
            ;;
        run)
            run_vm "$@"
            ;;
        shell)
            run_shell
            ;;
        clean)
            clean_image
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            print_error "Unknown command: $COMMAND"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

main "$@"
