"""
Example unit tests for sims_ci_pipe package
"""
import unittest
import desc.sims_ci_pipe

class sims_ci_pipeTestCase(unittest.TestCase):
    def setUp(self):
        self.message = 'Hello, world'

    def tearDown(self):
        pass

    def test_run(self):
        foo = desc.sims_ci_pipe.sims_ci_pipe(self.message)
        self.assertEqual(foo.run(), self.message)

    def test_failure(self):
        self.assertRaises(TypeError, desc.sims_ci_pipe.sims_ci_pipe)
        foo = desc.sims_ci_pipe.sims_ci_pipe(self.message)
        self.assertRaises(RuntimeError, foo.run, True)

if __name__ == '__main__':
    unittest.main()
