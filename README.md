# Sterowanie Magazynem

## Prerequisites
 * Anaconda
 https://www.anaconda.com/distribution/
 * github
 * git lfs
 
 https://git-lfs.github.com/

To download the repository use the github bash in the location you want the project
```
git clone https://github.com/MaczekO/Magazyn.git projectfolder
```


## Installation:
  * Python v. 3.7.4
    ```
    conda create -n yourenvname python=3.7.4 anaconda
    ```
    to use the environment
    ```
    conda activate yourenvname
    ```
  * Python packages from requirements.txt
    
    After activating the envioronment
    ```
    cd pojectPath
    pip install -r requirements.txt
    ```
  * Meteor v. 1.8.1
     
     To install meteor, you first have to install the chocolatey, follow the tutorial here:
     https://chocolatey.org/install
     
     After that open commandline and type in:
     ```
     choco install meteor
     ```
     Restart the command line andf go to the projectPath/meteor and create the project
     ```
     meteor create front
     ```
     
    * Meteor packages
    
      After creating the project type in the command line
      ```
      cd front
      meteor add accounts-ui accounts-password materialize:materialize@=0.97.0
      ```
  * After crating the project, place the files in it from the /meteor/front folder in respective places
  
## Running the app:

In order to run the app you need to launch meteor. To do that type in the commandline:
```
cd projectPath/meteor/front
meteor
```
And open your browser to go to the
```
http://localhost:3000/
```
