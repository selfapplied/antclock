# Zero-image μVM Makefile
# Builds the minimal executable kernel for CE system

CC = gcc
CFLAGS = -Wall -Wextra -O2 -std=c11 -pedantic
LDFLAGS = -lm

# Target binary
TARGET = zero_vm
TEST_TARGET = test_vm

# Source files
SRC = zero_vm.c
MAIN_SRC = main.c
HDR = zero_vm.h
OBJ = $(SRC:.c=.o)
MAIN_OBJ = $(MAIN_SRC:.c=.o)

# Test files
TEST_SRC = tests/test_vm.c
TEST_OBJ = $(TEST_SRC:.c=.o)

.PHONY: all clean test examples install

# Default target
all: $(TARGET)

# Build main VM binary
$(TARGET): $(OBJ) $(MAIN_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "✓ Built Zero-image μVM (~18 kB)"
	@ls -lh $(TARGET) | awk '{print "  Binary size:", $$5}'

# Build object files
%.o: %.c $(HDR)
	$(CC) $(CFLAGS) -c $< -o $@

# Build test suite
test: $(TARGET) $(TEST_TARGET)
	@echo "Running VM tests..."
	./$(TEST_TARGET)

$(TEST_TARGET): $(TEST_OBJ) zero_vm.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "✓ Built test suite"

# Build examples
examples: $(TARGET)
	@echo "Building example programs..."
	@cd examples && $(MAKE)
	@echo "✓ Examples built"

# Install VM system-wide (optional)
install: $(TARGET)
	@echo "Installing Zero-image μVM..."
	install -d $(DESTDIR)/usr/local/bin
	install -m 755 $(TARGET) $(DESTDIR)/usr/local/bin/
	@echo "✓ Installed to /usr/local/bin/"

# Clean build artifacts
clean:
	rm -f $(TARGET) $(TEST_TARGET) $(OBJ) $(MAIN_OBJ) $(TEST_OBJ)
	@cd examples && $(MAKE) clean 2>/dev/null || true
	@echo "✓ Cleaned build artifacts"

# Docker build
docker:
	@echo "Building Docker image..."
	docker build -t zero-vm:latest .
	@echo "✓ Docker image built"

# WASM build (requires emscripten)
wasm: $(SRC) $(HDR)
	@echo "Building WASM target..."
	emcc $(SRC) -o zero_vm.wasm -O2 -s WASM=1 -s EXPORTED_FUNCTIONS='["_vm_create","_vm_destroy","_vm_execute_opcode"]'
	@echo "✓ WASM target built"

# Print VM statistics
stats: $(TARGET)
	@echo "=== Zero-image μVM Statistics ==="
	@size $(TARGET)
	@echo ""
	@echo "Source lines:"
	@wc -l $(SRC) $(HDR)
	@echo ""
	@echo "Binary info:"
	@file $(TARGET)

# Help target
help:
	@echo "Zero-image μVM Build System"
	@echo ""
	@echo "Targets:"
	@echo "  all       - Build VM binary (default)"
	@echo "  test      - Build and run test suite"
	@echo "  examples  - Build example programs"
	@echo "  install   - Install VM system-wide"
	@echo "  clean     - Remove build artifacts"
	@echo "  docker    - Build Docker image"
	@echo "  wasm      - Build WASM target (requires emscripten)"
	@echo "  stats     - Show VM statistics"
	@echo "  help      - Show this help"
