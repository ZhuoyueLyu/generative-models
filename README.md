# Audio Interpolation and Translation Using Generative Models

Generative models such as Adversarially Constrained Autoencoder Interpolation (ACAI), Variational Autoencoder (VAE), and Generative Adversarial Networks (GAN) have proven to be powerful to generate new data. Our goals of this project are to (1) explore the applicability of ACAI on image and music interpolation with hand-written digits and MIDI notes datasets, (2) apply BiGAN and VAE on speech generation with Free Spoken Digit Dataset (FSDD) dataset, and (3) apply MelGAN-VC for voice translation that transfers audio styles from one to another on the FSDD dataset. Based on our experiments, the GANs performed better on audio generation than other autoencoder related models. 

## VAE.ipynb, BiGAN.ipynb \& MelGAN-VC.ipynb
These are the notebooks that contain the code for the respective models.

## MNIST \& MIDI 
The driver files that train the ACAI models are MNIST_ACAI.py or MIDI_ACAI under respective folders. 

The MIDI_parse.py won't run. It takes the clean_midi folder from the Lakh dataset and extract several piano notes from midi files. 
But the clean_midi folder is too large for github. 

