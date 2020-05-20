import unittest


class MyTestCase(unittest.TestCase):
    def test_the_outcome_was_expected(self):
        self.assertAlmostEqual(.1, .1)


if __name__ == '__main__':
    unittest.main()
