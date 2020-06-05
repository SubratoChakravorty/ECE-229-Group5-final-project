import os.path as path

__all__ = ['school_data_file', 'student_data_file', 'variables_file', 'vargroup_file', 'cache_dir', 'ui_assets_dir']

school_data_file = path.join(path.dirname(path.dirname(__file__)), 'data/school_data.csv')
student_data_file = path.join(path.dirname(path.dirname(__file__)), 'data/student_data.csv')
variables_file = path.join(path.dirname(path.dirname(__file__)), 'data/variables.csv')
vargroup_file = path.join(path.dirname(path.dirname(__file__)), 'data/var_group.json')
cache_dir = path.join(path.dirname(path.dirname(__file__)), 'cache/')
ui_assets_dir = path.join(path.dirname(path.dirname(__file__)), 'src/ui/assets')
