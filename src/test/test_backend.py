import unittest

import src.config as Config
import src.univariate_methods as um
import pandas as pd


# Here's an example test case to get you started
class TestDataFetching(unittest.TestCase):
    def test_return_fields_returns_a_dict(self):
        result = um.return_fields(Config.student_data_file)
        assert isinstance(result, dict)

    def test_get_counts(self):
        name = "X1SEX"
        result = um.get_counts(name, file_loc=Config.student_data_file)

        assert isinstance(result, pd.Series)
        assert result.name == name

    def test_get_field_data(self):
        name = "X1SES"
        result = um.get_field_data(field_name=name,
                                   file_loc=Config.student_data_file)
        assert isinstance(result, pd.Series)

    def test_get_binned_data(self):
        name = "X1SES"
        result = um.get_binned_data(field_name=name,
                                    file_loc=Config.student_data_file)

        assert isinstance(result, dict)
        assert ("range" in result) and ("count" in result)

    def test_get_hierarchical_data(self):
        fields = ["X1SEX", "X1RACE", "X1SES"]
        f, mean = um.get_hierarchical_data(fields=fields,
                                     file_loc=Config.student_data_file)

        assert isinstance(f, pd.DataFrame)
        assert isinstance(float(mean), float)

    def test_get_var_group(self):
        group = "group1"
        result = um.get_var_group(group, Config.vargroup_file)
        assert isinstance(result, list)

    def test_get_var_info(self):
        result = um.get_var_info(Config.variables_file)
        assert isinstance(result, pd.DataFrame)

    def test_get_stats(self):
        field = "X1SCIEFF"
        result = um.get_stats(field, Config.student_data_file)

        assert isinstance(result, tuple) and len(result) == 3
        assert isinstance(sum(result), float)


if __name__ == '__main__':
    unittest.main()
