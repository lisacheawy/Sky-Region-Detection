# Sky Region Detection

### Introduction
This is a sky region detection program that is able to extract regions of the sky from several datasets with varying landscapes and illumination. It employs contour validation and edge detection without the use of deep neural networks. The algorithm was tested on the following datasets which can be obtained from https://cs.valdosta.edu/~rpmihail/skyfinder/images/index.html :
* 1093, 4795, 8438 and 10870 during development 
* 4232, 8438, 9483 and 10917 during evaluation

The program is able to produce masks which display the extracted sky region, along with the average accuracy for each dataset. Conversely, the program will display the average false positives obtained for datasets which do not originally have a sky region. An average accuracy of 89.72% and 93.57% was obtained when running the algorithm on the development and evaluation datasets, respectively.
<br/><br/>
As night-time images were included in all datasets, it proves extremely difficult to extract the sky region from such scenarios as they are incorrectly composed and do not contain the information needed for extraction. This is caused by the heavy noise present on all night-time images which is a result of the camera's sensitivity to light, alongside non-uniform illumination in the image. Although continuous blurring could eventually remove the noise, it will substantially affect the clarity of contours and edges during extraction. Hence, the approach employed in this program will utilize the latest/closest day-time mask on the subsequent sequence of night-time images as it provides the most relevant sky region of that particular day.

### Output examples
| Original image | Mask produced | Ground truth  |
|    :----:      |    :----:     |    :----:     |
| <img src="https://user-images.githubusercontent.com/73222559/177741630-469aeb42-eb68-44cc-9cff-9af61f3cad67.jpg" width=300/>  | <img src="https://user-images.githubusercontent.com/73222559/177741658-71d57442-f56f-4663-a4e7-033cded4a564.png" width=300/> | <img src="https://user-images.githubusercontent.com/73222559/177741682-d6508cd3-dc35-44a9-a47e-cffd0577e20a.png" width=300/>  |
|<img src="https://user-images.githubusercontent.com/73222559/177740578-9668070d-2952-4d69-95fa-d8552788923b.jpg" width=300/>  | <img src="https://user-images.githubusercontent.com/73222559/177740977-3fcbfbd7-790b-45bb-9de8-e29e7afe2904.png" width=240/>        | <img src="https://user-images.githubusercontent.com/73222559/177741300-c5cf1bb1-239f-46ef-a002-67b74b84bb56.png" width=300/>  |
|<img src="https://user-images.githubusercontent.com/73222559/177742162-e2534471-7a01-4388-ba77-3e2504fd516f.jpg" width=300/>  | <img src="https://user-images.githubusercontent.com/73222559/177742471-6f7c4299-884b-446a-b0cc-e5b63aae4c67.png" width=300/>        | <img src="https://user-images.githubusercontent.com/73222559/177742196-7cfd1705-61ed-4e3c-b4f3-acbb0f4fb0f6.png" width=300/>  |


### Built with:
- Python (3.9)

### Libraries and packages used:
- OpenCV
- OS
- NumPy
- scikit-image

### Prerequisites
```
1. Install Anaconda with Python v3.9.
```

## Usage
The active directory of the program already contains the datasets and the corresponding `GroundTruth` used for evaluation and their output masks from the program. However, if you would like to reproduce the masks, you may delete all folders ending with `_mask` and rerun the program.
<br/><br/>
To test with other datasets, ensure that their folders are placed in the same directory as the `18111195.py` file. Also ensure that their ground truths are placed into the `GroundTruths` folder and labelled with their corresponding names.

```
1. From the Anaconda Navigator, launch the Spyder IDE.
2. Run the `18111195.py` file.
```
