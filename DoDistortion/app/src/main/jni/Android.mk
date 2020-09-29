LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

OpenCV_INSTALL_MODULES := on
OpenCV_CAMERA_MODULES := off

OPENCV_LIB_TYPE := SHARED

ifeq ("$(wildcard $(OPENCV_MK_PATH))","")
include $(LOCAL_PATH)/native/jni/OpenCV.mk
else
include $(OPENCV_MK_PATH)
endif

LOCAL_MODULE := DistortionCV

LOCAL_SRC_FILES := com_example_pynixs_dodistortion_OpenCVUtils.cpp

LOCAL_LDLIBS +=  -lm -llog -landroid -ljnigraphics



include $(BUILD_SHARED_LIBRARY)