from setuptools import setup, find_packages

setup(
    name='pupil-detection',
    version='5.2.0',
    description='pupil detection with n-back task prorgam',
    author='Kaixin JI',
    author_email='kaji@student.unimelb.edu.au',
    include_package_data=True,
    package_dir={'': 'client_app'},
    packages=[''],
    package_data={'': ['*.dat', 'tasks/*.txt', 'control.txt']},
    install_requires=[
        'numpy~=1.19.2',
    #     'setuptools~=40.8.0',
    #     'imutils~=0.5.3',
        'scipy~=1.5.2',
    #     'cmake~=3.18.2',
    #     'dlib~=19.17',
        'opencv-python~=4.4.0',
    ],
    py_modules = ['main', 'Task', 'client', 'client_proxyl', 'taskProgram'],
    entry_points={
        'console_scripts': [
            'papp=main:main'
        ]
    },
)