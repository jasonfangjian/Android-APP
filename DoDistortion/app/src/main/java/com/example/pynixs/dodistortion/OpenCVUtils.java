package com.example.pynixs.dodistortion;
import android.graphics.Bitmap;

public class OpenCVUtils {
    static {
        System.loadLibrary("DistortionCV");
    }
    public static native Bitmap getDistortion(Bitmap src,Bitmap frame);
}
