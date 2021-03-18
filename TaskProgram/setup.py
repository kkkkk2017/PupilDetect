from setuptools import setup, find_packages

setup(
    name='nback_prorgam',
    version='2.0.0',
    description='n-back task program - stimulus position not change',
    author='Kaixin JI',
    author_email='kaji@student.unimelb.edu.au',
    include_package_data=True,
    package_dir={},
    packages=[''],
    package_data={'': ['tasks/*.txt', 'tasks/*.mp4']},
    install_requires=[
        'opencv-python~=4.4.0',
    ],
    entry_points={
        'console_scripts': [
            'nback=main:main'
        ]
    },
)