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
  * Python v. 3.7.4 /w environment. run Anaconda prompt in project location then run
    ```
    conda env create -f environment.yml 
    ```
    to use the environment
    ```
    conda activate yourenvname
    ```
## Running the app:
  * To create the optimal solution for /data/orders.csv run the following code in the project folder
  ```
  python backend.py
  ```
  * To run the server application with generated data
  ```
  python Frontend/app.py
  ```
  You will be prompted about the port on which the server oparates (default localhost:8050)
  To login as an admin enter:
  login: Admin
  password: test123

