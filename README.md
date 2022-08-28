# NST-histological-images
## Project

This project consists into creating a histological slide by a stain-style transfer.
This can be done using a Neural Style Transfer network and using as style image a histological slide stained with hematoxylin-eosin and as content image a slide with no stain. 

The image were preprocessed in order to be coincident, so the only difference between the two image is the presence of the stain. 
This is useful when evaluating the result of the generated images, since in this way we are able to use the *Structural Similatiry Index*. 

## Table of contents

 

 - `images`
 - `NST.py`
 - `SSIM.py`
 - `requirements.txt`

The folder `images` contains all the images, in particular the ones used as input, which are titled *internal*, *ext* and *mid*, to distinguish the ROIs, with the adjective *colored* or  *white* based on whether the source image is stained or not. 
In this folder are also contained the generated images. 

The file `requirements.txt` contains packages useful for the the project.

`NST.py` is the Neural Style Transfer python script. 
  All the parameters are set in order to obtain the best result for the generated images. 

 `SSIM.py` calculate the Structural Index Similarity between the generated image and the original stained one. 
 ## Installation 
 To install the packages usefull for the project clone the repository and use pip: 
 ```
git clone https://github.com/francescapunzetti/NST-histological-images.git
cd NST-histological-images
pip install -r requirements.txt
```
In this way the whole repository in cloned.
## How to run the project
Firstly, using `NST.py` choose as input for the neural network a pair images labeled with *white* and *colored*, for example the two shown below:
<div align='center'>
<table cellspacing="2" cellpadding="2" width="600" border="0">
<tbody>
<tr>
<td valign="top" width="300"><img src="https://i.ibb.co/5kH6cN2/mid-colored.jpg" alt="mid-colored" align=”center” border="0"></a></td>
<td valign="top" width="300"><img src="https://i.ibb.co/W2V1yb7/mid-white.jpg" alt="mid-white" align=”center” border="0"></a></td>
</tr>
</tbody>
</table>
</div>

The output of the neural style transfer will be saved in the folder `images` .
The name associated to the generated image is the `number of epochs + 'epochs'.jpg`. 
There will be also a plot of the total loss in function of the number of iterations. 

After this, to evaluate the quality of the generated image, open `SSIM.py`, instert as `original` and `new` the name of the original stained slide and the name of the generated image and run the script. 
The output will be an image that show the two images in grey scale with the respective SSIM, as shown in the example:
<div align='center'>
<img src="https://i.ibb.co/kcWfxBg/mid.png" align="middle" alt="mid" width="300" border="0">
</div>
