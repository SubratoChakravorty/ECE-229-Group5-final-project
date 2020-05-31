from setuptools import setup

# todo: finish or delete this
setup(
   name='SSE',
   version='1.0',
   description="A dashboard for analysis of students' science-self efficacy",
   author='Ian Pegg, Subrato Chakravorty, Yan Sun, Daniel You, Heqian Lu, Kai Wang',
   author_email='wangjinjie722@gmail.com',
   packages=['src'], install_requires=
   ['scikit-learn',
   "dash>=1.4",
   "pandas>=1.0",
   "plotly>=4.6",
   "pytest==5.4.2",
   "selenium>=3",
   "scikit-learn>=0.22",
   "flask",
   "pytest-dash",
   "dash_bootstrap_components",
   "sphinx==1.8.5",
   "sphinx_rtd_theme==0.4.2",
   "recommonmark",
   ]  #same as name
)