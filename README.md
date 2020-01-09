# Plant_Disease_Classification

# Dependency

Required libraries and packages can be download from requirements.txt

`pip install -r requirements.txt`

# Dataset:

we have used plant village dataset consisting of 38 class plant diseases described below
1. Apple___Apple_scab
2. Apple___Black_rot
3. Apple___Cedar_apple_rust
4. Apple___healthy
5. Blueberry___healthy
6. Cherry_(including_sour)___Powdery_mildew
7. Cherry_(including_sour)___healthy
8. Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot
9. Corn_(maize)___Common_rust_
10. Corn_(maize)___Northern_Leaf_Blight
11. Corn_(maize)___healthy
12. Grape___Black_rot
13. Grape___Esca_(Black_Measles)
14. Grape___Leaf_blight_(Isariopsis_Leaf_Spot)
15. Grape___healthy
16. Orange___Haunglongbing_(Citrus_greening)
17. Peach___Bacterial_spot
18. Peach___healthy
19. Pepper,_bell___Bacterial_spot
20. Pepper,_bell___healthy
21. Potato___Early_blight
22. Potato___Late_blight
23. Potato___healthy
24. Raspberry___healthy
25. Soybean___healthy
26. Squash___Powdery_mildew
27. Strawberry___Leaf_scorch
28. Strawberry___healthy
29. Tomato___Bacterial_spot
30. Tomato___Early_blight
31. Tomato___Late_blight
32. Tomato___Leaf_Mold
33. Tomato___Septoria_leaf_spot
34. Tomato___Spider_mites Two-spotted_spider_mite
35. Tomato___Target_Spot
36. Tomato___Tomato_Yellow_Leaf_Curl_Virus
37. Tomato___Tomato_mosaic_virus
38. Tomato___healthy

# Neural Network

we have checked 6 neural network architectures for training the model and got nearly 95-99% accuracy.
1. LeNet
2. MiniVGGNet
3. AlexNet
4. MiniGoogLeNet
5. GoogleNet
6. ResNet

# training network
We used `train.py` to train the network. This file takes three command line arguments --train is the path to the input train images --test is the path to the input test images --model is the path to output model

to run `train.py`, insert following command

`python train.py --train datasets/train --test datasets/val --model output/alexnet.hdf5`

# testing network
We used `predict.py` to test the network. This file takes two command line arguments --model is the path to pretraied input model --image is the path to the image for which we want to predict disease.

to run `predict.py`, insert following command

`python predict.py --model output/alexnet.hdf5 --image imagepath`
