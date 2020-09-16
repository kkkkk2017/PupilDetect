from setuptools import setup, find_packages

setup(
    name='pupil-detection',
    version='2.0.0',
    description='pupil detection with n-back task prorgam',
    author='Kaixin JI',
    author_email='kaji@student.unimelb.edu.au',
    packages=find_packages(),
    # install_requires=[
    #     'numpy~=1.16.2',
    #     'setuptools~=40.8.0',
    #     'imutils~=0.5.3',
    #     'scipy~=1.2.1',
    #     'dlib~=19.17',
    #     'opencv-python~=3.2.0',
    # ],
    py_modules = ['main', 'Task', 'client', 'client_proxyl', 'taskProgram'],
    entry_points={
        'console_scripts': [
            'app=client.main:run'
        ]
    },
    package_data={
        'client': ['*.dat']
    },
)