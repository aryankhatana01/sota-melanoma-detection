# sota-melanoma-detection
Implementation of state of the art Melanoma Detection Model using PyTorch.

A demo can be found [Here](https://drive.google.com/file/d/12bNm0sFw0kl935zYHM9wJ13M9ViJ0Di-/view?usp=sharing)

Steps to run the code : 

0. Open your terminal and ```cd``` into this directory.
1. First install the requirements using ```pip3 install -r requirements.txt```
2. Downloads the weights from [Here](https://www.kaggle.com/datasets/aryankhatana/sotamelanoma) and place them in the ```best_models``` folder.
3. Type ```chmod +x start_server.sh``` to give execution permission to the bash file.
4. Type ```./start_server.sh``` to start the backend API.
5. ```cd``` into the ```frontend``` folder and type ```npm run start``` to run the React frontend.
