#ID2image: Leakage of non-ID information into face descriptors and inversion from descriptors to images
we present an optimisation strategy that uses a StyleGAN generative model to  recover images from an ID embedding.

## Requirments
- pytorch = 0.10.0
- Download the CelebA dataset from [https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)


## Model Zoo
- The models are available for non-commercial research purposes only.  
- All models can be found in here.  
    - Rregressors for Non-ID attribute prediction of CelebA: [GDrive](https://drive.google.com/drive/folders/15bqQhs3_yiUYwkP8lp3vFz9fkHKHsLPU) (the traning code can be found in this repository)
    - Pretrained VGGface and VGGface2 model from [https://www.robots.ox.ac.uk/~albanie/pytorch-models.html](https://www.robots.ox.ac.uk/~albanie/pytorch-models.html)
    - Dlib 68-points Face landmark model from [http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
    
## Usage
Take an face image you'd like to inversed. Then you align and crop the face image in an VGG Face style.
Then you can inversion this face image from a face descriptor by using the ID_encode_to_image.py script.
```bash
python ID_encode_to_image.py 
"test_image.jpg" 
--save_file_name inversion_result # The folder name use to save the result image and plot at.
--learning_rate 1 
--weight_landmarkloss 0.0006 
--vgg_to_hist 0.01 # weight of histogram loss
--iterations 2500  
```

### ID_encode_to_image.py
```
arguments:

  image_path            Filepath of the image to be inversed.
  
  --learning_rate       Learning rate for SGD.
                        
  --model_type          the model to use from InterFaceGAN repo.

  --iterations          Number of optimizations steps.
  
  --latent_type         type of StyleGAN latent vector (Z or WP), 
  
  --vggface_to_latent   Whether or not to use a vgg_to_latent model to intialise StyleGAN latent
  
  --vgg_to_hist         Whether or not to use histogram_loss during inversion
  
  --weight_landmarkloss the weighr of landmark_loss in loss fucntion
  
  --weight_histloss     the weighr of histogram_loss in loss fucntion
  
  --save_file_name      the folder used to save results
```
## Non-ID attribute prediction from ID
We begin by exploring to what extent we are able to esti-mate non-ID “attributes” from an ID descriptor provided by a pretrained face encoder CNN. We use “attribute” here in very general terms,  including image-based attributes suchas  landmark  positions  and  colour  histograms  and  non-IDface attributes such as the presence or absence of a smile,glasses  or  hat. 
