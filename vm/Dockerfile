# Zero-image μVM Docker Container
# Minimal container for running the CE spectral virtual machine

FROM alpine:latest

# Install build dependencies
RUN apk add --no-cache gcc musl-dev make

# Set working directory
WORKDIR /vm

# Copy source files
COPY zero_vm.h zero_vm.c main.c Makefile ./

# Build the VM
RUN make clean && make

# Create volume mount point for programs
RUN mkdir /programs

VOLUME ["/programs"]

# Set entrypoint
ENTRYPOINT ["./zero_vm"]

# Default help command
CMD ["--help"]

# Metadata
LABEL org.opencontainers.image.title="Zero-image μVM"
LABEL org.opencontainers.image.description="Minimal executable kernel for CE1 bracket algebra"
LABEL org.opencontainers.image.version="0.1.0"
LABEL org.opencontainers.image.source="https://github.com/selfapplied/antclock"
LABEL org.opencontainers.image.licenses="CC-BY-SA-4.0"
