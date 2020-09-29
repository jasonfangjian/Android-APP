# Android-APP

## Main Purpose
The project is developing an Android application which can realized correcting Keystone Distortion in Digital Images. For example, when we put a rectangle document like a paper or a textbook on the desk. And then we use our phone or camera to take a photo of this document at an angle. Not right in front of the document. Then the document in the photo will looks like a keystone. This is due to the principle of perspective, also called perspective distortion. What we need to do is to restore the file that looks like a keystone in this photo to a normal rectangle, which is somewhat like scanning. 

## Files
[PerspectiveDistortion](./PerspectiveDistortion): algorithm for automatically correcting Keystone distortion of digital images (Trained HED neural network model for edge detection, using C++ for Image enhancement, calling OpenCV for correction).<br>

[DoDistortion](./DoDistortion): Developed an Android APP by Java (with JNI) and TensorFlow Lite to implement the above algorithm.

[CorrectDistortion](./CorrectDistortion.apk): Demo .apk file for trial.

[assets](./assets): test pictures

## Demo
