from setuptools import setup

setup(
    name='d6tjoin',
    version='0.2.1',
    packages=['d6tjoin'],
    url='https://github.com/d6t/d6tjoin',
    license='MIT',
    author='DataBolt Team',
    author_email='support@databolt.tech',
    description='Easily join python pandas dataframes',
    long_description='Easily join python pandas dataframes'
        'See https://github.com/d6t/d6tjoin for details',
    install_requires=[
        'numpy',
        'pandas',
        'jellyfish',
        'joblib',
        'd6tstack'
    ],
    include_package_data=True,
    python_requires='>=3.6'
)
