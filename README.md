<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>
<!--
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->



<!-- ABOUT THE PROJECT -->
## About The Project

[![Product Name Screen Shot][product-screenshot]]()

In neural decoding research, one of the most intriguing topics is the reconstruction of perceived 
natural images based on fMRI signals. We used publicly available Natural Scenes Dataset 
benchmark and renovate BrainDiffuser with another Regression is Batch-MultiOutput regression, `email`, `project_title`, `project_description`, `project_license`
-
<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With

* [![VDVAE][VDVAE]][openai/vdvae]
* [![VD][VersatileDiffusion]]SHI-Labs/Versatile-Diffusion]
* [![Regression][Batch-MultiOutput Regression]][[regression]

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

 You have to create environment by checking requirements yourself by by entering `conda env create -f environment.yml`.

### Data Installation

1. Download NSD data from NSD AWS Server:
   ```sh
   cd data
   python download_nsddata.py
   ```
   This file code will automatically download subj01 for the FMRI of the first person of dataset.
2. Prepare NSD data for the Reconstruction Task:
   ```sh
   cd data
  python prepare_nsddata.py -sub 1
  python prepare_nsddata.py -sub 2
  python prepare_nsddata.py -sub 5
  python prepare_nsddata.py -sub 7
   ```
### Reconstruction with VDVAE

1. Download pretrained VDVAE 
   ```sh
  wget https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets-2/imagenet64-iter-1600000-log.jsonl
wget https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets-2/imagenet64-iter-1600000-model.th
wget https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets-2/imagenet64-iter-1600000-model-ema.th
wget https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets-2/imagenet64-iter-1600000-opt.th
   ```
2. Extract VDVAE latent features of stimuli images for any subject 'x' using:
   ```sh
   python scripts/vdvae_extract_features.py -sub 1
   ```
3. Train regression models from fMRI to VDVAE latent features and save test predictions using:
   ```sh
   python scripts/vdvae_regression.py -sub 1
   ```
4. Reconstruct images from predicted test features using
   ```sh
   python scripts/vdvae_reconstruct_images.py -sub x
   ```
<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Reconstruction with Versatile Diffusion
Reconstruct images from predicted test features using `python scripts/versatilediffusion_reconstruct_images.py -sub 1` .
### Reconstruction with Versatile Diffusion



<!-- LICENSE -->
## License

Distributed under the project_license. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

TRANTHUY - ) - tranthuy2810@gmail.com
VUTUQUYNH - ) - vtq06011@gmail.com

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- References -->
## References

* Codes in vdvae directory are derived from [openai/vdvae]
* Codes in versatile_diffusion directory are derived from earlier version of [SHI-Labs/Versatile-Diffusion]
* Dataset used in the studies are obtained from [Natural Scenes Dataset]

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[openai/vdvae]: https://github.com/openai/vdvae
[SHI-Labs/Versatile-Diffusion]: https://github.com/SHI-Labs/Versatile-Diffusion
[Natural Scenes Dataset]: https://naturalscenesdataset.org/
[regression]: https://arxiv.org/abs/2403.19421
[product-screenshot]: https://github.com/TranThuy28/BrainToImage/blob/main/results/ssRESULT.png
