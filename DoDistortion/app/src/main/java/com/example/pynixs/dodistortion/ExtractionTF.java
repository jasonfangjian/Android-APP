package com.example.pynixs.dodistortion;


import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.graphics.Matrix;
import android.media.ThumbnailUtils;
import android.util.Log;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

public class ExtractionTF{
    private static final String TAG="ExtractionTF";
    // 设置模型输入输出维度
    private static final int IN_COL = 256;
    private static final int IN_ROW = 256;
    private static final int OUT_COL = 256;
    private static final int OUT_ROW = 256;
    private static final int IMAGE_SIZE = 256;
    //模型中输入变量的名称
    private static final String inputName = "hed_input:0";
    //模型中输出变量的名称
    private static final String outputName = "hed/dsn_fuse/conv2d/BiasAdd:0";

    private static final String IsTrainingName = "is_training:0";
    TensorFlowInferenceInterface inferenceInterface;
    static {
        //加载libtensorflow_inference.so库文件
        System.loadLibrary("tensorflow_inference");
        Log.e(TAG,"libtensorflow_inference.so库加载成功");
    }

    ExtractionTF(AssetManager assetManager, String modePath) {
        //初始化TensorFlowInferenceInterface对象
        inferenceInterface = new TensorFlowInferenceInterface(assetManager,modePath);
        Log.e(TAG,"TensoFlow模型文件加载成功");
    }

    /**
     *  利用训练好的TensoFlow模型预测结果
     * @param bitmap 输入被测试的bitmap图
     * @return 返回预测结果，int数组
     */
    public float[] getPredict(Bitmap bitmap) {
        Bitmap newBitmap=zoomBitmap(bitmap,IMAGE_SIZE ,IMAGE_SIZE );//将图片缩放带256*256
        float[] inputdata =  getPixels(newBitmap);
        //将数据feed给tensorflow的输入节点
        inferenceInterface.feed(inputName, inputdata,1,IN_COL, IN_ROW,3);
        boolean[] bTrans=new boolean[]{false};
        inferenceInterface.feed(IsTrainingName,bTrans);
        //运行tensorflow
        String[] outputNames = new String[] {outputName};
        inferenceInterface.run(outputNames);
        ///获取输出节点的输出信息
        float[] outputs = new float[OUT_COL*OUT_ROW]; //用于存储模型的输出数据
        inferenceInterface.fetch(outputName, outputs);
        return outputs;
    }


    private float[] getPixels(Bitmap bitmap) {


        int[] intValues = new int[IMAGE_SIZE * IMAGE_SIZE];
        float[] floatValues = new float[IMAGE_SIZE * IMAGE_SIZE * 3];

        if (bitmap.getWidth() != IMAGE_SIZE || bitmap.getHeight() != IMAGE_SIZE) {
            // rescale the bitmap if needed
            bitmap = ThumbnailUtils.extractThumbnail(bitmap, IMAGE_SIZE, IMAGE_SIZE);
        }

        bitmap.getPixels(intValues,0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        for (int i = 0; i < intValues.length; ++i) {
            final int val = intValues[i];
            floatValues[i * 3] = Color.red(val) / 255.0f;
            floatValues[i * 3 + 1] = Color.green(val) / 255.0f;
            floatValues[i * 3 + 2] = Color.blue(val) / 255.0f;
        }
        return floatValues;
    }

    public static Bitmap zoomBitmap(Bitmap bitmap, int w, int h){
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        Matrix matrix = new Matrix();
        float scaleWidth = ((float) w / width);
        float scaleHeight = ((float) h / height);
        matrix.postScale(scaleWidth, scaleHeight);
        Bitmap newBmp = Bitmap.createBitmap(bitmap, 0, 0, width, height,
                matrix, true);
        return newBmp;
    }
}