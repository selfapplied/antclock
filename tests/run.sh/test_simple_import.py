"""
Simple test to verify PYTHONPATH works
"""

def main():
    try:
        # Try to import something that should be available in the project root
        import sys
        project_root = "/Users/joelstover/antclock"
        if project_root in sys.path:
            print("SUCCESS: PYTHONPATH correctly includes project root!")
        else:
            print(f"FAILED: Project root not in sys.path: {sys.path}")
    except Exception as e:
        print(f"FAILED: {e}")

if __name__ == "__main__":
    main()
