Pupil Detection Program
--
##Installation Guide
##### Step 1, if you don't have miniconda installed, download miniconda for python3.7 version from this website. 
            https://docs.conda.io/en/latest/miniconda.html

##### Step 2, go to your command-prompt and run. (Python version is restricted on python3.7).
        conda create --name pupil_env python=3.7
        
##### Step 3, activate your conda environment with
        conda activate pupil_env

#### Step 4, run the following command to install cmake
        conda install cmake==3.18.2
                
##### Step 5, navigate to the directory/folder where you saved the pupil-detection.tar.gz is. (No need to unzip this package!)
        For example, you saved in Downloads folder. C:/Downloads> pip install [filename].

##### Step 6, run the following command to install the package
    pip install pupil-detection-6.0.0.tar.gz [or pupil-detection-6.0.0.tar.gz if you got the 6.0.0 version]  
    
##### Step 7, if the installation succeed, run 'papp' to run the program. 
   If not, check your environments and see if it contains the pupil-detection and all of the required packages(listed below). 
   If it does not contains, the installation might not succeed, repeat the previous steps.
   
   ###### The required packages are
        python~=3.7
        cmake~=3.18.2
        numpy~=1.19.2
        setuptools~=40.8.0
        imutils~=0.5.3
        scipy~=1.5.3
        dlib~=19.17
        opencv-python~=4.4.0


### Till now, the installation all done.


##### Step 8, click on the Calibration Button first to finish the calibration. 
    When calibrating eyes, press 'y' if the shown image has correctly circuled your pupil, otherwise, press 'n' or space.
    When you fnished, press 'q' to close the window.
    
##### Step 9, click on the 'Start' Button to start the program and n-back task program. The server program must runs beforehand.
