"""
Tests for AntClock runtime module.

This module tests the AntRuntime class and CE-layer decorators.
"""

import unittest
from antclock.runtime import AntRuntime, make_ce_decorators


class TestAntRuntime(unittest.TestCase):
    """Test cases for AntRuntime class."""

    def test_init_default(self):
        """Test AntRuntime initializes with default values."""
        runtime = AntRuntime()
        self.assertEqual(runtime.x, 1.0)
        self.assertEqual(runtime.phase, 0.0)
        self.assertEqual(runtime.A, 0)
        self.assertEqual(runtime.log, [])
        self.assertIsNotNone(runtime.walker)

    def test_init_custom_params(self):
        """Test AntRuntime initializes with custom parameters."""
        runtime = AntRuntime(x_0=2.0, chi_feg=0.5)
        self.assertEqual(runtime.x, 2.0)
        self.assertEqual(runtime.walker.chi_feg, 0.5)

    def test_tick_advances_A(self):
        """Test that tick() advances the experiential time A."""
        runtime = AntRuntime()
        initial_A = runtime.A
        runtime.tick(event_type="bracket", layer=1)
        self.assertGreater(runtime.A, initial_A)

    def test_tick_logs_entry(self):
        """Test that tick() appends a log entry."""
        runtime = AntRuntime()
        runtime.tick(event_type="bracket", layer=1)
        self.assertEqual(len(runtime.log), 1)
        entry = runtime.log[0]
        self.assertIn('A', entry)
        self.assertIn('x', entry)
        self.assertIn('phase', entry)
        self.assertIn('digit_shell', entry)
        self.assertIn('clock_rate', entry)
        self.assertIn('event_type', entry)
        self.assertIn('layer', entry)

    def test_tick_with_state(self):
        """Test that tick() logs state when provided."""
        runtime = AntRuntime()
        runtime.tick(state={"test": "data"})
        entry = runtime.log[0]
        self.assertIn('state', entry)
        self.assertEqual(entry['state'], {"test": "data"})

    def test_tick_with_metadata(self):
        """Test that tick() merges metadata into log entry."""
        runtime = AntRuntime()
        runtime.tick(metadata={"custom_field": 123})
        entry = runtime.log[0]
        self.assertIn('custom_field', entry)
        self.assertEqual(entry['custom_field'], 123)


class TestCEDecorators(unittest.TestCase):
    """Test cases for CE-layer decorators."""

    def setUp(self):
        """Set up a fresh runtime and decorators for each test."""
        self.runtime = AntRuntime()
        self.antce1, self.antce2, self.antce3 = make_ce_decorators(self.runtime)

    def test_antce1_basic(self):
        """Test antce1 decorator basic functionality."""
        @self.antce1()
        def test_func():
            return "result"

        result = test_func()
        self.assertEqual(result, "result")
        self.assertGreater(self.runtime.A, 0)
        self.assertEqual(len(self.runtime.log), 1)
        entry = self.runtime.log[0]
        self.assertEqual(entry['layer'], 1)
        self.assertEqual(entry['event_type'], "bracket")

    def test_antce2_basic(self):
        """Test antce2 decorator basic functionality."""
        @self.antce2()
        def test_func():
            return "flow_result"

        result = test_func()
        self.assertEqual(result, "flow_result")
        self.assertEqual(len(self.runtime.log), 1)
        entry = self.runtime.log[0]
        self.assertEqual(entry['layer'], 2)
        self.assertEqual(entry['event_type'], "flow_step")

    def test_antce3_basic(self):
        """Test antce3 decorator basic functionality."""
        @self.antce3()
        def test_func():
            return "simplex_result"

        result = test_func()
        self.assertEqual(result, "simplex_result")
        self.assertEqual(len(self.runtime.log), 1)
        entry = self.runtime.log[0]
        self.assertEqual(entry['layer'], 3)
        self.assertEqual(entry['event_type'], "simplex_flip")

    def test_antce1_logs_return_value(self):
        """Test antce1 logs return value as state by default."""
        @self.antce1()
        def test_func():
            return {"data": 42}

        test_func()
        entry = self.runtime.log[0]
        self.assertIn('state', entry)
        self.assertEqual(entry['state'], {"data": 42})

    def test_antce3_logs_args(self):
        """Test antce3 logs args as state by default."""
        @self.antce3()
        def test_func(a, b, c=3):
            return a + b + c

        test_func(1, 2, c=3)
        entry = self.runtime.log[0]
        self.assertIn('state', entry)
        args, kwargs = entry['state']
        self.assertEqual(args, (1, 2))
        self.assertEqual(kwargs, {'c': 3})

    def test_log_state_none(self):
        """Test log_state='none' does not log state."""
        @self.antce1(log_state="none")
        def test_func():
            return "should not be logged"

        test_func()
        entry = self.runtime.log[0]
        self.assertNotIn('state', entry)

    def test_attach_A_true(self):
        """Test attach_A=True injects A into kwargs."""
        @self.antce2(attach_A=True)
        def test_func(A=None):
            return A

        # First call - A should be 0 (before tick increments it)
        result = test_func()
        self.assertEqual(result, 0)

    def test_attach_A_does_not_override(self):
        """Test attach_A=True does not override caller-provided A."""
        @self.antce2(attach_A=True)
        def test_func(A=None):
            return A

        result = test_func(A=999)
        self.assertEqual(result, 999)

    def test_metadata_fn(self):
        """Test metadata_fn adds custom metadata."""
        def custom_metadata(result, args, kwargs):
            return {"result_length": len(result)}

        @self.antce1(metadata_fn=custom_metadata)
        def test_func():
            return "hello"

        test_func()
        entry = self.runtime.log[0]
        self.assertIn('result_length', entry)
        self.assertEqual(entry['result_length'], 5)

    def test_metadata_fn_exception_fails_closed(self):
        """Test metadata_fn exception does not break the decorator."""
        def broken_metadata(result, args, kwargs):
            raise ValueError("Intentional error")

        @self.antce1(metadata_fn=broken_metadata)
        def test_func():
            return "works"

        # Should not raise
        result = test_func()
        self.assertEqual(result, "works")
        # Entry should be logged without extra metadata
        self.assertEqual(len(self.runtime.log), 1)

    def test_multiple_decorated_calls(self):
        """Test multiple decorated function calls accumulate correctly."""
        @self.antce1()
        def func1():
            return 1

        @self.antce2()
        def func2():
            return 2

        @self.antce3()
        def func3():
            return 3

        func1()
        func2()
        func3()

        self.assertEqual(len(self.runtime.log), 3)
        self.assertEqual(self.runtime.log[0]['layer'], 1)
        self.assertEqual(self.runtime.log[1]['layer'], 2)
        self.assertEqual(self.runtime.log[2]['layer'], 3)


def main():
    """Run all tests."""
    unittest.main()


if __name__ == "__main__":
    main()
