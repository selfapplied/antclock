"""
Simple test to verify PYTHONPATH works
"""

def main():
    try:
        # Try to import something that should be available in the project root
        import sys
        import os

        # Get the directory containing this test file, then go up two levels to project root
        # This works regardless of where the test is run from
        test_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(test_dir))

        if project_root in sys.path:
            print("SUCCESS: PYTHONPATH correctly includes project root!")
            print(f"Project root: {project_root}")
        else:
            print(f"FAILED: Project root not in sys.path")
            print(f"Expected project root: {project_root}")
            print(f"Current sys.path: {sys.path}")
    except Exception as e:
        print(f"FAILED: {e}")

if __name__ == "__main__":
    main()
