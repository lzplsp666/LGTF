This implementation is for the single-GPU configuration. All experiments can be reproduced on a GPU with more than 10GB memory (e.g., 1080Ti)!

### Environment 
The code is tested on PyTorch 1.13.1.

### Datasets 

We suggest downloading all datasets to a root directory (`${data_root}`), and renaming the directory of each dataset as suggested in `${ID_to_DIRNAME}` in `./data/datautils.py`. This would allow you to evaluate multiple datasets within the same run.     
If this is not feasible, you could evaluate different datasets separately, and change the `${data_root}` accordingly in the bash script.


For zero/few-shot classification, we consider 11 datasets:
* [ImageNet](https://image-net.org/index.php) 
* [Flower102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz)
* [DTD](https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz)
* [OxfordPets](https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz)
* [StanfordCars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)
* [UCF101](https://drive.google.com/file/d/10Jqome3vtUA2keJkNanAiFpgbyC9Hc2O/view?usp=sharing)
* [Caltech101](http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz)
* [Food101](http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz)
* [SUN397](http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz)
* [Aircraft](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz)
* [EuroSAT](http://madm.dfki.de/files/sentinel/EuroSAT.zip)
 
For out-of-distribution generalization, we consider 4 datasets:

* [ImageNet-A](https://github.com/hendrycks/natural-adv-examples)
* [ImageNet-R](https://github.com/hendrycks/imagenet-r)
* [ImageNet-V2](https://s3-us-west-2.amazonaws.com/imagenetv2public/imagenetv2-matched-frequency.tar.gz)
* [ImageNet-Sketch](https://github.com/HaohanWang/ImageNet-Sketch)

## Run LGTF
testsets=I/Flower102/DTD/Pets/Cars/UCF101/Caltech101/Food101/SUN397/Aircraft/eurosat

root  --test_sets datasetname(such as Flower102) --selection_p 0.1 -a ViT-B/16 -b 32  
--ctx_init a_photo_of_a --memory_size 50 --text_prompt tip_cupl  
--log camera_ready_dmn_searched_vit --gpu 1 --n_shot 16 --n_augview 0   --beta 5.5   
--use_searched_param  --num_important_channel 0 --lambda_ape 0.3 --epoch 20 --lr 0.001 --ft
### Searched optimal classifier weights of LGTF for different task settings and datasets with the VITB/16 backbone.


| Settings   | Items | ImageNet | SUN   | Aircraft | EuroSAT | Cars  | Food  | Pets  | Flower | Caltech | DTD   | UCF   |
|------------|-------|----------|-------|----------|---------|-------|-------|-------|--------|---------|-------|-------|
| **8shot**  | α     | 0.7      | 1.0   | 1.0      | 1.0     | 1.0   | 1.0   | 0.7   | 0.9    | 1.0     | 1.0   | 1.0   |
|            | γ     | 0.8      | 0.75  | 1.7      | 0.45    | 0.85  | 0.35  | 0.3   | 0.15   | 0.6     | 1.0   | 1.15  |
| **16shot** | α     | 1.0      | 0.9   | 0.9      | 0.7     | 1.0   | 1.0   | 0.9   | 1.0    | 1.0     | 0.8   | 0.9   |
|            | γ     | 1.9      | 1.05  | 0.65     | 0.3     | 0.8   | 0.45  | 0.4   | 0.7    | 0.75    | 0.25  | 0.55  |
| **8shot Results** | 74.35 | 76.43 | 43.05    | 83.9    | 82.61 | 86.44 | 92.78 | 97.36  | 96.84   | 71.16 | 85.09 |    avg:80.91
| **16shot Results**| 75.98 | 78.3  | 48.03    | 87.98   | 85.85 | 87.45 | 93.84 | 98.36  | 96.84   | 74.70 | 86.7  |        83.09

If you want to test  robustness and generation, you can use the provided weights to test on ImageNet-derived datasets. The weights and caches are located in the folders “a” and v2 respectively. In the `get_llm_text_features(self, dataset, n_shot)` function within `clip/fixclip`, only retain the line `self.llm_local_text = torch.load('llm_local_text.pt')`.


attention:
in clip/fix_clip   Please change it to your own path link.
folder_path='/data/gpt_file'
result = ' '.join(dataset)+'.json'
fold_path='/data/gpt4_data'
res=' '.join(dataset)+'.json'

The data in the GPT4 folder needs to be copied from the GPT folder.