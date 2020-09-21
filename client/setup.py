from setuptools import setup, find_packages

setup(
    name='pupil-detection',
    version='2.0.0',
    description='pupil detection with n-back task prorgam',
    author='Kaixin JI',
    author_email='kaji@student.unimelb.edu.au',
    include_package_data=True,
    packages=find_packages(),
    package_data={'client.client_app': ['*.dat']},
    install_requires=[
        'numpy~=1.16.6',
        'setuptools~=40.8.0',
        'imutils~=0.5.3',
        'scipy~=1.2.1',
        'dlib~=19.17',
        'opencv-python~=3.4.8.29',
        'cmake~=3.18.2',
    ],
    py_modules = ['main', 'Task', 'client', 'client_proxyl', 'taskProgram'],
    entry_points={
        'console_scripts': [
            'papp=client_app.main:main'
        ]
    },
)