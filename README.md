# GENSON #
<b>Generate JSON</b>

### Make JSON file for Roboflow ###
* now: version 0.2.0

## Process
1. Original
2. Binary
3. Resize (1440x1024)
4. Center Crop (512x512)
5. Bilateral Filter
6. Dilation
7. Semantic Segmentation
8. Contour Detection


## Model
> You can download the trained UNET model.
>> Through the below link.  
>> Or move to download tab.  
> It should be located under the ckpt folder.  
```bash
https://drive.google.com/file/d/14Caw6ubdZgsAyFc5xCpxVhOH_yspXPMD/view?usp=sharing
```


## Make exe file
```bash
conda create -n seah python=3.9 -y
conda activate seah
pip install -r requirements.txt
pyinstaller -F -w main_genson.py -n Genson.exe
```