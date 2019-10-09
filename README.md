############ N-Glycan-image-detection ###################

Bioinformatics research project at Georgetown University

Author: Nhat T. Duong
Email: ntd12@georgetown.edu

Objective:
Process image N-glycan image and output them as glycan sequences.

Update:
To be update:
- Develop technique to measure color gradient on a grapth and merge potential difference (normalize colors)  
- Perform output on amounts of monosaccharide types and match them up with publication database to outlines potential error
- Potential magic parameters can be determine by machine learning ?

Previous build/log:
- Build 10/9/2019: linedetecttest2 module contain findLines(img) function take image(cv2) parameter to highlight all lines found in the png input
- Build 10/5/2019: outline symbols representing chemical bond between monosaccharide and output name of the symbol(using overlaying technique)
- Build 10/1/2019: outline rectangle around contour of shapes; perform bitwise matching of two image picked up from countours finder.
- Build 9/28/2019: self-develop method to perform bitwise matching of shape png, method to perform Thresholding.
- 9/26/2019: learn about openCV, image processing, pixels, iteration of pixels


Resources:
Main wiki page: https://edwardslab.bmcb.georgetown.edu/glycandata/Main_Page
Data sources: https://github.com/glygen-glycan-data/PyGly/tree/master/smw/glycandata/export
  - Samples image pull from images-extended.zip
  - Image can be pair with GlycO derived enzyme annotation and publications on unicarbkb.


References:
Project is Recommended by Dr. Edward Nathan (https://gufaculty360.georgetown.edu/s/contact/00336000014RcUqAAK/nathan-edwards)
Dr. Edward's lab (https://edwardslab.bmcb.georgetown.edu/)
Dr. Ross (https://www.linkedin.com/in/karen-ross-671207a3/)
