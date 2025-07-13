# Task-2: Face Generation

The goal of this task is to develop a face generation model that takes face-embeddings from a pre-trained model as an input and produces a human face. The desired output is a 3x128x128 image of a human face. The maximum time allowed for the training is 6 Hrs.

## Approach taken

1. **Data Preparation**: The dataset used is a mixture of WiderFace test dataset and a openly available [Face-Detection-Dataset](https://www.kaggle.com/datasets/fareselmenshawii/face-detection-dataset/data) from Kaggle.  This dataset is then passed to the YOLOv11 model trained to detect faces in [Task-1](https://github.com/Akhil-Theerthala/Task1-Face-Detection-Rust). These detected faces are then filtered to be at least 128x128 pixels in size and then resized to 128x128 pixels.
    - The dataset was generated in two phases:
      - In the first phase, a smaller portion of the wider-face dataset was chosen, filtered to remove the images that were too similar (as can be seen from the Task-1 repository), and then they were filtered based on a minimum presence of 90.5x90.5 pixles (since the target size is 128x128, a 128/sqrt(2) =90.5 px would be the minimum size to ensure that the face details are not lost in the upscaling). This phase contained around 4100 training images, which were enough to estimate the direction of the GAN training.
      - In the second phase, the dataset was expanded to include more images from the LFW Validation set and the Kaggle dataset mentioned above, expanding the dataset size to 16k images. Though this is not a large dataset in comparison to CelebA (which generates a interpretable face after 15-20 epochs), it was the maximum that my compute resources could handle.
2. **Pre-trained Model**: A pre-trained [TinyVit model](https://huggingface.co/gaunernst/vit_tiny_patch8_112.arcface_ms1mv3) trained on MS1MV3 dataset is used to extract the face embeddings. The model outputs a [1, 512] vector for each face, and is small enough to run instantly.
3. **Compute Used**: The training is done mainly on two devices
    - **Kaggle P100**: Kaggle's free provision of 30hrs of 16GB GPU compute was used to train the diffusion models.
    - **Apple Silicon-M4**: The GAN model was trained on Apple Silicon M4 chip which has a 16GB Unified Memory. The training was done using PyTorch with MPS backend.
4. **Approches taken**:
    - **Diffusion Model**: An unconditional and conditional diffusion models are trained to generate faces from the face embeddings. Around 3-4 different experiments were done, through this approach, but couldn't be explored further due to the training time limits and the compute resources available.
    - **GAN Experiments**: A GAN model was used to generate faces from the face embeddings. The overall experiments can be classified into three phases:
      - **Phase-1**: Discriminator dominated the training, making the generator produce close to random noise as the outputs.
      - **Phase-2**: The balance between the generator and discriminator was achieved, but the outputs still weren't different from random noise. However, some qualitative results show that a sillouette of a face is being generated.****
      - **Phase-3**: Dataset expansion phase. The earlier two phases were trained on a small dataset of around 4000 images. The dataset in this phase was expanded to around 15-16k images. Then, two generators were trained with to provide **detailed** analysis.
      - **Gray-scale training**: After training the "Final-run" of the face generation model specified, another experment was done to train the model to generate gray-scale features. This was done to see if the model can learn the features of a face better if it were to be in gray-scale. 

## Results

- The `diffusion_model_training.ipynb` contains the code used to train the diffusion models, and the 4-5 experiments that were initially done. Because of compute restrictions and time constraints, the diffusion model approach was not explored further and the GAN model was chosen as the primary approach.
- The `faceGAN` folder contains the code used to train the GAN model, and the results of the training, with the `trainer.py` file being the main entry point for training the GAN model.
- The results of both the approaches were not satisfactory. Though the diffusion model was clearly able to generate faces, due to the compute and time constraints, the model couldn't be trained for a longer time to generate more realistic faces.
- The GAN model was suitable to be trained on local compute, but the results were not satisfactory, as the model requires either a larger dataset or a more complex architecture that takes too long to generate faces.
- The limitations are further backed by the fact that the performance of the unconditional models in the same timeframe were able to generate similar but slightly better faces.
- Further experiments can be done to improve the results, but the current time and compute constraints would have to be relaxed to achieve better results.
- The results of the inference can be seen in the `inference_visualization.ipynb` file, which contains the code to visualize the generated faces from the GAN model.

## Model Devlopment History

- The GAN model initially contained a simple deconvolutional generator and a fastvit discriminator. However, since the discrimnator was too powerful, it dominated the training after very few steps, making the generator feedback loop ineffective.
- Hence the discriminator was replaced with a simpler CNN based architecture, which was further tuned to attain a balance between the generator and discriminator. It was also during this process that the batch normalization from the discrimator was replaced with spectral normalization, which theoretically produces a better discriminator.
- After the discriminator balance was attained, the generator still was producing random noise as the output even after 40-50 epochs of training. This showed that the generator was too simple to learn the features of a face. Hence the next few experiments involved improving the generator architecture, which included:
  - Increasing the scale of the layers
  - Replacing the Transposed Convolution layers with Upsample blocks
  - Using Softplus loss instead of the traditional BCE loss, which was shown to produce better stable results in the GAN training.
- These changes were able to produce a stable interplay between the generator and discriminator, and also push the generator to at least produce a "silloutette" of a face, which often included bright patches that resembled the face location similar to the input images (though no actual features of a face).
- This architecutre was then trained through an Unconditional training on high-quality sampled (~8k) CELEBA dataset, to establish a baseline on how good the model can perform in an ideal scenario. **This baseline was used to establish a hypothesis that the model cannot perform well on the current dataset size and complexity.** This can be verified by the WandB dashboards titled "Unconditional-Baseline-1" and "Unconditional-Baseline-2" in the WandB project linked below.
- Based on this fact, a reasonable expansion of the 4k dataset was done to 16k, and the final training run was done. The final training run was done in two phases, with the same generator and discriminator architecture, but with different scale of the generators. The first phase used a smaller inital channel expansion of 512, while the second phase used a larger initial channel expansion of 1024.
- The final generations are available in the `inference_visualization.ipynb` file.

## LLM Usage

- Coming from pure NLP experience, I used ChatGPT-o3 as a brainstorming assistant, to fill my personal knowledge gaps in the field of computer vision. Most of the discussion was around how the GAN model can be improved on the current problem, specifically moving from a simple deconvolution and batchnormalization to a use better Upsample blocks and spectral norms. The discussion also included the benefits of Softplus loss over the traditional BCE loss, and how it can help the better problem.
- The entire discussion and brainstorming can be found in this [record](https://chatgpt.com/share/6873623c-8a78-800b-88fb-b098b47373ae).

## Observations and Evaluations

- **GAN Training Losses**: Monitored via WandB, showing discriminator and generator losses over time (see [WandB dashboard](https://wandb.ai/silvervein/face-generation-gan-mps?nw=nwuserakhiltvsn)).
- **Qualitative Evaluation**: Images generated after every few epochs are available in the dashboard and `inference_visualization.ipynb`. The final images show approximate face silhouettes but no distinct facial features
- **Quantitative Scores**: For the sake of a quantitative evaluation, a validation set of 1100 faces was used. The quantitative measurements were based on the cosine similarity between the generated face embeddings and the original face embeddings, trying to see if any of the generated faces contain some features of a face.
- Though the cosine similarity scores are not ideal as they are between two different faces, but they can act as a proxy measure of how close the generated face is to a real face. If they are too close, then that means the generated image is an exact copy of the original image. However, since some noise is introduced in the generation process, this can be avoided. For the current model, the average cosine similarity is 0.15, which means that the generated images are very different from the faces in the validation set.
- Other metrics like FID and KID were not used as the jupyter kernel was repeatedly crashing while calculating these metrics. 

## WandB dashboards and Image samples

- The WandB project details for the GAN model can be found in this [link](https://wandb.ai/silvervein/face-generation-gan-mps?nw=nwuserakhiltvsn). This project dashboards include the training losses of the generator and discriminator, the generated images for every few epochs during training, the system charts.
- The sample images generated by the GAN model can be seen in the same dashboard.
