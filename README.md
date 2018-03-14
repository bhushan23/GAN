# GAN

I am experimenting with different known GAN architecture and also trying out on new GAN ideas

## AutoEncoders
  * Basic AutoEncoder on MNIST
  
    ![AutoEncoder Output](https://github.com/bhushan23/GAN/blob/master/AutoEncoder/genImg/image_9.png)

## VAE 
  * Variational AutoEncoder on MNIST 
  * Extracting Mean and Variance from same output layer works best
  * Useful tutorial - https://jaan.io/what-is-variational-autoencoder-vae-tutorial/
  
## Vanilla GAN
  * Simple Fully connected Generative Adversarial Network
  
    ![Vanilla GAN](https://github.com/bhushan23/GAN/blob/master/Vanilla%20GAN/Vanilla_Gan.gif) 
    
    ![LossGraph](https://github.com/bhushan23/GAN/blob/master/Vanilla%20GAN/Loss_Plot_Vanilla_Min_GAN_200.png)
  
## Co-Operative GAN on top of Vanilla GAN
  * Multiple Generators are trained simultaneously and one performing better is choosen and copied by other Generators for next     iteration.
  * Idea is to Co-Operatively improve over a period of time
  * Best Performing Generator can be with Min/Max Loss or can be choosen randomly
  * With Min Loss:
     
     ![Vanilla Min_GAN](https://github.com/bhushan23/GAN/blob/master/Co-Operative-GAN/Min/Min_Multiple_Gens.gif)
     
     ![Loss](https://github.com/bhushan23/GAN/blob/master/Co-Operative-GAN/Min/Loss_Plot_Vanilla_Min_Co-Operative_GAN_200.png)
  * With Max Loss:
      
      ![Vanilla Max GAN](https://github.com/bhushan23/GAN/blob/master/Co-Operative-GAN/Max/Max_Multiple_Gens.gif)
      
      ![Loss](https://github.com/bhushan23/GAN/blob/master/Co-Operative-GAN/Max/Loss_Plot_Vanilla_Max_Co-Operative_GAN_200.png)
  * With Random selection:
      
      ![Vanilla Random GAN](https://github.com/bhushan23/GAN/blob/master/Co-Operative-GAN/Random/Random_Multiple_Gens.gif)
      
      ![Loss](https://github.com/bhushan23/GAN/blob/master/Co-Operative-GAN/Random/Loss_Plot_Vanilla_Random_Co-Operative_GAN_200.png)
  * With Min Loss but Generators input is same noise:
      
      ![Vanilla Min loss same noise](https://github.com/bhushan23/GAN/blob/master/Co-Operative-GAN/Min_D_Once/Min_Multiple_Gens.gif)
      
      ![loss](https://github.com/bhushan23/GAN/blob/master/Co-Operative-GAN/Min_D_Once/Loss_Plot_Vanilla_Min_Same_Noise_Co-Operative_GAN_200.png)
      
  ## DC-GAN
   Under Implementation
