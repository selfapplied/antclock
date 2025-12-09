"""
Zero-image μVM Python Bindings

Provides a Python interface to the Zero-image μVM using ctypes.
This allows Python code to create and execute VM programs programmatically.

Example:
    >>> from python_bindings import ZeroVM, Opcode
    >>> vm = ZeroVM()
    >>> vm.push(1.5 + 2.0j, depth=5, monodromy=0.5)
    >>> vm.execute_opcode(Opcode.PROJECT)
    >>> state = vm.get_state()
    >>> print(f"Antclock: {state['antclock']}")
"""

import ctypes
from ctypes import c_void_p, c_uint32, c_uint64, c_uint8, c_double, c_float, c_bool, c_char_p
from enum import IntEnum
from pathlib import Path
from typing import Optional, Dict, Any
import os


class Opcode(IntEnum):
    """VM Opcodes corresponding to CE1 bracket algebra"""
    PROJECT = 0x10  # {} - Project tensor path to spectral value
    DEPTH   = 0x20  # [] - Compute p-adic distance
    MORPH   = 0x30  # () - Apply Hecke action
    WITNESS = 0x40  # <> - Extract witness tuple


class ZeroVM:
    """
    Python interface to the Zero-image μVM
    
    This class wraps the C VM implementation and provides
    a Pythonic interface for creating and executing VM programs.
    """
    
    def __init__(self, lib_path: Optional[str] = None):
        """
        Initialize the VM wrapper
        
        Args:
            lib_path: Path to the compiled VM shared library.
                     If None, tries to find it automatically.
        """
        if lib_path is None:
            # Try to find the library
            lib_path = self._find_library()
        
        # Load the shared library
        self.lib = ctypes.CDLL(lib_path)
        
        # Define function signatures
        self._setup_function_signatures()
        
        # Create VM instance
        self.vm = self.lib.vm_create()
        if not self.vm:
            raise RuntimeError("Failed to create VM")
    
    def _find_library(self) -> str:
        """Find the compiled VM library"""
        # Look for zero_vm.so in the vm directory
        vm_dir = Path(__file__).parent
        candidates = [
            vm_dir / "libzero_vm.so",
            vm_dir / "zero_vm.so",
            vm_dir / "libzero_vm.dylib",
            vm_dir / "zero_vm.dylib",
            vm_dir / "zero_vm.dll",
        ]
        
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        
        raise FileNotFoundError(
            "Could not find VM library. Build it first with: cd vm && make"
        )
    
    def _setup_function_signatures(self):
        """Setup ctypes function signatures"""
        # vm_create() -> VMState*
        self.lib.vm_create.argtypes = []
        self.lib.vm_create.restype = c_void_p
        
        # vm_destroy(VMState*)
        self.lib.vm_destroy.argtypes = [c_void_p]
        self.lib.vm_destroy.restype = None
        
        # vm_push(VMState*, complex, uint32_t, float)
        # Note: For complex numbers, we pass real and imaginary separately
        self.lib.vm_push.argtypes = [c_void_p, c_double, c_double, c_uint32, c_float]
        self.lib.vm_push.restype = None
        
        # vm_execute_opcode(VMState*, Opcode) -> bool
        self.lib.vm_execute_opcode.argtypes = [c_void_p, c_uint32]
        self.lib.vm_execute_opcode.restype = c_bool
        
        # vm_load_program(VMState*, const char*) -> bool
        self.lib.vm_load_program.argtypes = [c_void_p, c_char_p]
        self.lib.vm_load_program.restype = c_bool
        
        # vm_run(VMState*)
        self.lib.vm_run.argtypes = [c_void_p]
        self.lib.vm_run.restype = None
    
    def push(self, rho: complex, depth: int, monodromy: float):
        """
        Push a spectral value onto the VM stack
        
        Args:
            rho: Complex spectral value
            depth: Ultrametric depth
            monodromy: Monodromy angle in radians
        """
        self.lib.vm_push(
            self.vm,
            c_double(rho.real),
            c_double(rho.imag),
            c_uint32(depth),
            c_float(monodromy)
        )
    
    def execute_opcode(self, opcode: Opcode) -> bool:
        """
        Execute a single VM opcode
        
        Args:
            opcode: The opcode to execute
            
        Returns:
            True if execution succeeded
        """
        return self.lib.vm_execute_opcode(self.vm, c_uint32(opcode))
    
    def load_program(self, filename: str) -> bool:
        """
        Load a program from a Zero-image file
        
        Args:
            filename: Path to the program file
            
        Returns:
            True if loading succeeded
        """
        return self.lib.vm_load_program(self.vm, filename.encode('utf-8'))
    
    def run(self):
        """Run the loaded program to completion"""
        self.lib.vm_run(self.vm)
    
    def __del__(self):
        """Cleanup VM when object is destroyed"""
        if hasattr(self, 'vm') and self.vm:
            self.lib.vm_destroy(self.vm)
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if hasattr(self, 'vm') and self.vm:
            self.lib.vm_destroy(self.vm)
            self.vm = None


# Example usage
if __name__ == "__main__":
    print("Zero-image μVM Python Bindings Demo")
    print("=" * 50)
    
    # Note: This requires building the VM as a shared library
    # Run: gcc -shared -o libzero_vm.so -fPIC zero_vm.c -lm
    
    try:
        with ZeroVM() as vm:
            print("✓ VM created successfully")
            
            # Push some values
            vm.push(1.5 + 2.0j, depth=5, monodromy=0.5)
            print("✓ Pushed spectral value: 1.5 + 2.0j")
            
            vm.push(3.0 + 4.0j, depth=3, monodromy=1.0)
            print("✓ Pushed spectral value: 3.0 + 4.0j")
            
            # Execute DEPTH opcode
            success = vm.execute_opcode(Opcode.DEPTH)
            if success:
                print("✓ Executed DEPTH opcode")
            
            print("\nDemo completed!")
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nTo use Python bindings, build the VM as a shared library:")
        print("  cd vm")
        print("  gcc -shared -o libzero_vm.so -fPIC zero_vm.c -lm")
    except Exception as e:
        print(f"Error: {e}")
