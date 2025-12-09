# Examples Makefile

CC = gcc
CFLAGS = -Wall -Wextra -O2 -std=c11 -I..
LDFLAGS = -lm

EXAMPLES = simple_projection distance_computation morphism_chain witness_extraction guardian_demo

.PHONY: all clean

all: $(EXAMPLES)

simple_projection: simple_projection.o ../zero_vm.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

distance_computation: distance_computation.o ../zero_vm.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

morphism_chain: morphism_chain.o ../zero_vm.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

witness_extraction: witness_extraction.o ../zero_vm.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

guardian_demo: guardian_demo.o ../zero_vm.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(EXAMPLES) *.o

.PHONY: run
run: all
	@echo "Running examples..."
	@echo ""
	@./simple_projection
	@echo ""
	@./distance_computation
	@echo ""
	@./morphism_chain
	@echo ""
	@./witness_extraction
	@echo ""
	@./guardian_demo
