# Sky Region Detection

### Introduction
This is a sky region detection program that is able to extract regions of the sky from several datasets with varying landscapes and illumination. It employs contour validation and edge detection without the use of deep neural networks. The algorithm was tested on the following datasets which can be obtained from https://cs.valdosta.edu/~rpmihail/skyfinder/images/index.html :
* 1093, 4795, 8438 and 10870 during development 
* 4232, 8438, 9483 and 10917 during evaluation
The program is able to produce masks which display the extracted sky region, along with the average accuracy for each dataset. Conversely, the program will display the average false positives obtained for datasets which do not originally have a sky region. An average accuracy of 89.72% and 93.57% was obtained when running the algorithm on the development and evaluation datasets, respectively.
<br/><br/>
As night-time images were included in all datasets, it proves extremely difficult to extract the sky region from such scenarios as they are incorrectly composed and do not contain the information needed for extraction. This is caused by the heavy noise present on all night-time images which is a result of the camera's sensitivity to light, alongside non-uniform illumination in the image. Although continuous blurring could eventually remove the noise, it will substantially affect the clarity of contours and edges during extraction. Hence, the approach employed in this program will utilize the latest/closest day-time mask on the subsequent sequence of night-time images as it provides the most relevant sky region of that particular day.

### Built with:
- Python (3.9)

### Libraries and packages used:

- OpenCV
- OS
- NumPy
- scikit-image
