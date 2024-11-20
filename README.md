The project is about detecting and counting the number of instance of the parijat in the give picture or the frame.

### Dataset Description

Dataset contained about 117 images containing the Night Jasmine(Parijat in Nepali) containing single or multiple instances in single photo. The data was agumented after the anotation, three times the original by decreasing brightness, saturation and contrast getting the number of instances of the class to be nearly 2000. The dataset was annotated using the CVAT. 

## Sample of dataset

<img src="dataset_sample.jpg" alt="drawing" width="300"/>

All the trained files can be seen in trained_files folder

## Running the Code
To run the code install the necesssary pacakges using 

` pip install -r requirements.txt ` 

Then run to User Interface 

`streamlit run app.py`

## Demo Video
![Demo Video](https://www.youtube.com/watch?v=JMre-All2ys)

**P.S**: The UI might not work properly for the camera for now.
