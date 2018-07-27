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
  * Multiple Generators are trained simultaneously and one performing better is choosen and copied by other Generators for next iteration.
  * Idea is to Co-Operatively improve over a period of time
  * Best Performing Generator can be with Min/Max Loss or can be choosen randomly
  * Winner takes all strategy
 
## Results
  ### With Min Loss:
![Vanilla Min_GAN](https://github.com/bhushan23/GAN/blob/master/Co-Operative-GAN/Min/Min_Multiple_Gens.gif)
<img src="https://github.com/bhushan23/GAN/blob/master/Co-Operative-GAN/Min/Vanilla%20GAN%20LossLoss_Plot_Vanilla_GAN_200.png" width="500">
  ### With Max Loss:
![Vanilla Max GAN](https://github.com/bhushan23/GAN/blob/master/Co-Operative-GAN/Max/Max_Multiple_Gens.gif)
![Loss](https://github.com/bhushan23/GAN/blob/master/Co-Operative-GAN/Max/Loss_Plot_Vanilla_Max_Co-Operative_GAN_200.png)
  ### With Random selection:
![Vanilla Random GAN](https://github.com/bhushan23/GAN/blob/master/Co-Operative-GAN/Random/Random_Multiple_Gens.gif)
![Loss](https://github.com/bhushan23/GAN/blob/master/Co-Operative-GAN/Random/Loss_Plot_Vanilla_Random_Co-Operative_GAN_200.png)
  ### With Min Loss but Generators input is same noise:
![Vanilla Min loss same noise](https://github.com/bhushan23/GAN/blob/master/Co-Operative-GAN/Min_D_Once/Min_Multiple_Gens.gif)
![loss](https://github.com/bhushan23/GAN/blob/master/Co-Operative-GAN/Min_D_Once/Loss_Plot_Vanilla_Min_Same_Noise_Co-Operative_GAN_200.png)
  ### Min vs Max vs Random (200th epoch output)
|![200th Epoch Image Min](https://github.com/bhushan23/GAN/blob/master/Co-Operative-GAN/Min/genImg/image_199.png)|![200th Epoch Image Max](https://github.com/bhushan23/GAN/blob/master/Co-Operative-GAN/Max/genImg/image_199.png)|![200th Epoch Image Random](https://github.com/bhushan23/GAN/blob/master/Co-Operative-GAN/Random/genImg/image_199.png)|
|:---:|:---:|:---:|
|*Min*|*Max*|*Random*|

### Why Max loss out performs Min loss
 * All generators loss is eventually decreasing
 * Taking Max everytime avoids trapping in saddle point and local minima as highe learning rate configuration will help come out of it
 * Taking max is safer approach to eventually reach to final solution

## Why Min loss does not work well enough
 * If one generator collapses, then it will generate minimum loss and fails

 
  
      
  ## DC-GAN
   * Implemented for CelebA dataset: [ComputerVision Repo](https://github.com/bhushan23/Computer-Vision/tree/master/CelebA)
