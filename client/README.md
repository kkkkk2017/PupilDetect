Pupil Detection Program
--
##Installation Guide
##### Step 1, if you don't have miniconda installed, download miniconda for python3.7 version from this website. 
            https://docs.conda.io/en/latest/miniconda.html

##### Step 2, go to your command-prompt and run.
        conda create --name pupil_env python=3.7
        
##### Step 3, activate your conda environment with
        conda activate pupil_env

##### Step 4, run the following command to install imutils, cmake, dlib (cmake needs to be install prior than dlib)
        conda install imutils
        conda install cmake
        conda install dlib
                
##### Step 5, navigate to the directory/folder where you saved the pupil-detection.tar.gz is. (No need to unzip this package!)
        For example, you saved in Downloads folder. C:/Downloads> pip install [filename].

##### Step 6, run the following command to install the package
    pip install pupil-detection-5.2.0.tar.gz   
    
##### Step 7, if the installation succeed, run 'papp' to run the program. 
   * If not, check your environments and see if it contains the pupil-detection and all of the required packages(listed below). 
   If it does not contains, the installation might not succeed, repeat the previous steps.
   * To manually install py-opencv, the command is 'conda install opencv-python' or 'pip install opencv-python'.
   ###### The required packages are
        python~=3.7.8
        cmake~=3.18.2
        numpy~=1.16.6
        setuptools~=40.8.0
        imutils~=0.5.3
        scipy~=1.5.2
        dlib~=19.21.0
        py-opencv~=4.4.0
    
 ###### if you received a error about Tkinter, check you python version by 
    conda list
 ###### if it is not python2.7, then you will need to remove the enviornment by
    conda env remove -n pupil_env
 then repeat from step 2. :(

### Till now, the installation all done.


##### Step 8, click on the Calibration Button first to finish the calibration. 
    When calibrating eyes, press 'y' if the shown image has correctly circuled your pupil, otherwise, press 'n' or space.
    When you finished, press 'q' to close the window.
    
##### Step 9, click on the 'Start' Button to start the program and n-back task program.
        On the top menu bar, button 'Prac' for start the practice task,
        'Start Task' for start the current task and 'Next' is for go the the next task,
        and 'rest' for start the resting time
        
        (The error label only display on the practise stage to help to familiar with the rule, 
        you can attend as many time as you want)
        
        * Remember click the 'Next' button to go to the next task. 
        