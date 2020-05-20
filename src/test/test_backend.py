import unittest

from src.config import *
import src.univariate_methods as um


# Here's an example test case to get you started
class TestDataFetching(unittest.TestCase):
    def test_return_fields_returns_a_dict(self):
        result = um.return_fields(student_data_file)
        assert isinstance(result, dict)


if __name__ == '__main__':
    unittest.main()
