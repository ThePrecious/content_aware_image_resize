### Content aware image resize
Have you ever tried to post your vacation pic on social media and find that it somehow crops a part of the image to fit its requirement, thereby cropping off a non trivial part of the image?? I did.. My aim in this project to build a deep learning application that can resize the image smartly, without cropping off the important part of the image. Fancy name for this project would be **Content aware image resizing.** :wink:

Currently you will find MVP of this project here. Wanna try? :smiley:

1. Goto - http://ec2-54-212-97-143.us-west-2.compute.amazonaws.com:5000/

2. Paste a link to the image you want to resize

3. Click on resize button

Wait for less than a minute :hourglass_flowing_sand: - drum roll .... here's your resized image :tada: :tada:.

Happy with the output of MVP??? Thank you. 
Not Happy with the output?? I am working on improving the model. There is a laundry list of todos. :woman_technologist:
Will update the model when it improves. Watch out this space.

**Note** - If the above link does not work (ie. If you get the message - "This site can’t be reached")- it means that I have turned off the ec2 instance to avoid unnecessary expense. :money_with_wings: 


**So, what is happening behind the scenes?**:female_detective:

I have trained a model with a bunch of images which I have resized using seam carving technique. Seam carving in simple words would be - a technique that identifies the path of pixels with low energy called seam and remove it to reduce the size or insert it to increase the size of the image.

So training input - is an image and training label - is seam carved image.

In this project I am focusing on reducing the size (384x400). In future, I would like to build a model to increase the size of the image too. Wikipedia does a good job of explaining seam carving - https://en.wikipedia.org/wiki/Seam_carving

I decided to use the famous u-net architecture for this problem. Consider the seams to be like segments in the image that the model has to learn about ie, the model should be able to identify low energy paths in the image.

The current model is not bad and not great. It can be improved and I am on it.

**To do -**
1. Improve the current model.
2. Build a model that can increase the size of the image
3. Add feature to the front end where user can select increase size or decrease size. Based on user input select the model.
