#!/run.sh
# AntClock-Self: 1764901313
"""
Test script with broken shebang to trigger fallback logic
"""
print("This should trigger the fallback logic!")
print("When run directly, it will fail with an error.")
print("When run with run.sh, it will be upgraded to a more specific shebang.")
print("Except this is a test script and we test with --dry-run=direct-self.")