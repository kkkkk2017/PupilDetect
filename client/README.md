Pupil Detection Program
--
##Installation Guide
##### Step 1, if you don't have miniconda installed, download miniconda for python2.7 version from this website. 
            https://docs.conda.io/en/latest/miniconda.html

##### Step 2, go to your command-prompt and run. (Python version is restricted on python2.7).
        conda create --name pupil_env python=2.7
        
##### Step 3, activate your conda environment with
        conda activate pupil_env
                
##### Step 4, go to the directory/folder where you saved the pupil-detection.tar.gz is. (No need to unzip this package!)
        For example, you saved in Downloads folder. C:/Downloads> pip install [filename].

##### Step 5, run the following command to install the package
    pip install pupil-detection-2.0.1.tar.gz [or pupil-detection-2.0.2.tar.gz if you got the 2.0.2 version]  
    
##### Step 6, if the installation succeed, run 'papp' to run the program. 
   If not, check your environments and see if it contains the pupil-detection and all of the required packages(listed below). 
   If it does not contains, the installation might not succeed, repeat the previous steps.
   
   ###### The required packages are
        python~=2.7
        cmake~=3.18.2
        numpy~=1.16.6
        setuptools~=40.8.0
        imutils~=0.5.3
        scipy~=1.2.1
        dlib~=19.17
        opencv-python~=3.2.0
    
** if you have problem with installing dlib or it shows a error about cmake, it might because you need to install cmake first.
    run 'conda install cmake==3.18.2'. 
    then repeat step 5 again.

### Till now, the installation all done.


##### Step 7, click on the Calibration Button first to finish the calibration. 
    When calibrating eyes, press 'y' if the shown image has correctly circuled your pupil, otherwise, press 'n' or space.
    When you fnished, press 'q' to close the window.
    
##### Step 8, click on the 'Start' Button to start the program and n-back task program. The server program must runs beforehand.

##### Step 9, when you finished all the task, wait for the server allow you to quit, then click 'Quit' to terminate the program. 