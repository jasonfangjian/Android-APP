# Android-APP

## Main Purpose
The project is developing an Android application which can realized correcting Keystone Distortion in Digital Images. For example, when we put a rectangle document like a paper or a textbook on the desk. And then we use our phone or camera to take a photo of this document at an angle. Not right in front of the document. Then the document in the photo will looks like a keystone. This is due to the principle of perspective, also called perspective distortion. What we need to do is to restore the file that looks like a keystone in this photo to a normal rectangle, which is somewhat like scanning. 

## Files
[PerspectiveDistortion](./PerspectiveDistortion): algorithm for automatically correcting Keystone distortion of digital images (Trained HED neural network model for edge detection, using C++ for Image enhancement, calling OpenCV for correction).<br>

[DoDistortion](./DoDistortion): Developed an Android APP by Java (with JNI) and TensorFlow Lite to implement the above algorithm.

[CorrectDistortion](./CorrectDistortion.apk): Demo .apk file for trial.

[assets](./assets): test pictures

## Demo
* Download the CorrectionDistortion.apk file on an Android smart phone.
* Installing the application on the phone.
* Open the APP and start trial.

1. The first button is to take the photo, and the second button is to start the process of correction.
<div align=center><img width="270" height="540" src="https://github.com/jasonfangjian/Android-APP/blob/master/assets/test1.jpg"/></div>

2. Taking the photoes.

<div align=center><img width="270" height="540" src="https://github.com/jasonfangjian/Android-APP/blob/master/assets/takephoto.jpg"/></div>

3. Test 1: Picture on the left is the photo before processing, and the picture on the right is the result after correction.

<div align=center><img width="270" height="540" src="https://github.com/jasonfangjian/Android-APP/blob/master/assets/test1.jpg"/><img width="270" height="540" src="https://github.com/jasonfangjian/Android-APP/blob/master/assets/result1.jpg"/></div>

4. Test 2: Picture on the left is the photo before processing, and the picture on the right is the result after correction.

<div align=center><img width="270" height="540" src="https://github.com/jasonfangjian/Android-APP/blob/master/assets/test2.jpg"/><img width="270" height="540" src="https://github.com/jasonfangjian/Android-APP/blob/master/assets/result2.jpg"/></div>

5. Test 3: Picture on the left is the photo before processing, and the picture on the right is the result after correction.

<div align=center><img width="270" height="540" src="https://github.com/jasonfangjian/Android-APP/blob/master/assets/test3.jpg"/><img width="270" height="540" src="https://github.com/jasonfangjian/Android-APP/blob/master/assets/result3.jpg"/></div>

