Pupil Detection Program
--
#### Step 1, on your command-line, run command 'python' and check if your python version.
            The python version for this program must be python2.7

#### Step 2, check if you have pip installed. If not, please follow this link and install pip.
    Windows: https://phoenixnap.com/kb/install-pip-windows
    Mac: https://www.geeksforgeeks.org/how-to-install-pip-in-macos/#:~:text=Download%20and%20Install%20pip%3A,directory%20as%20python%20is%20installed.&text=and%20wait%20through%20the%20installation,now%20installed%20on%20your%20system 

#### Step 3, go to the directory/folder where you save the pupil-detection.tar.gz is.

#### Step 4, run 'pip install pupil-detection-2.0.1.tar.gz [or pupil-detection-2.0.2.tar.gz]'    
    The required packages 
    python ~= 2.7
    numpy~=1.16.6
    setuptools~=40.8.0
    imutils~=0.5.3
    scipy~=1.2.1
    dlib~=19.17
    opencv-python~=3.2.0
    cmake~=3.18.2
    
** if it shows a error about cmake, it means that you need to install cmake on your computer.
   Please refer to this link about how to install cmake on your computer. 
   https://medium.com/analytics-vidhya/how-to-install-dlib-library-for-python-in-windows-10-57348ba1117f#:~:text=Now%20we%20can%20install%20dlib,need%20to%20install%20CMake%20library.&text=Then%2C%20you%20can%20install%20dlib%20library%20using%20pip%20install%20.&text=After%20passing%20enter%2C%20you%20laptop,run%20the%20C%2C%20C%2B%2B%20Compiler.

### Step 5, if the installation succeed, run 'papp' to run the program. 
   If not, check your environments and see if there is the pupil-detection and other required packages. If not, the installation might not succeed, repeat the previous steps. 

#### Step 6, click on the Calibration Button first to finish the calibration. 
    When calibrating eyes, press 'y' if the shown image has correctly circuled your pupil, otherwise, press 'n' or space.
    When you fnished, press 'q' to close the window.
    
#### Step 7, click on the 'Start' Button to start the program and n-back task program. The server program must be running beforehand.

#### Step 8, when you finished all the task, click 'Quit' to terminate the program.