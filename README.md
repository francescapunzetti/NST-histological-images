# NST-histological-images
## Project

This project consists into creating a histological slide by a stain-style transfer.
That's can be done using a Neural Style Transfer network and using as style image a histological slide stained with hematoxylin-eosin and as content image a slide with no stain. 

The image were preprocessed in order to be coincident, so the only difference between the two image is the presence of the stain. 
This is useful when evaluating the result of the generated images, because in that's way is possible to use the *Structural Similatiry Index*. 

## Table of contents

 

 - `images`
 - `NST.py`
 - `SSIM.py`

The folder `images` contains all the images, in particular the ones used as input, which are titled *internal*, *ext* and *mid*, to distinguish the ROIs, with the adjective *colored* or  *white* based on whether the source image is stained or not. 
In this folder are also contained the generated images. 

`NST.py` is the Neural Style Transfer python script. 
  All the parameters are set in order to obtain the best result for the generated images. 

 `SSIM.py` calculate the Structural Index Similarity between the generated image and the original stained one. 
 
## How to use the project
First download `NST.py` 
<div align='center'>
<table cellspacing="2" cellpadding="2" width="600" border="0">
<tbody>
<tr>
<td valign="top" width="300"><a href="https://imgbb.com/"><img src="https://i.ibb.co/5kH6cN2/mid-colored.jpg" alt="mid-colored" align=”center” border="0"></a></td>
<td valign="top" width="300"><a href="https://imgbb.com/"><img src="https://i.ibb.co/W2V1yb7/mid-white.jpg" alt="mid-white" align=”center” border="0"></a></td>
</tr>
</tbody>
</table>
</div>
