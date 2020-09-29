package com.example.pynixs.dodistortion;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.Environment;
import android.os.StrictMode;
import android.provider.MediaStore;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.widget.Button;
import android.widget.TextView;
import android.widget.ImageView;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.view.View;
import android.widget.Toast;

import java.io.File;
import java.io.IOException;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = "MainActivity";
    private static final String MODEL_FILE = "file:///android_asset/hed_graph.pb"; //模型存放路径

    private static final int REQUEST_EXTERNAL_STORAGE = 1;
    private static String[] PERMISSIONS_STORAGE = {
            "android.permission.READ_EXTERNAL_STORAGE",
            "android.permission.WRITE_EXTERNAL_STORAGE" };


    // Used to load the 'native-lib' library on application startup.
    static {
        System.loadLibrary("native-lib");
    }

    TextView txt;
    TextView tv;
    ImageView imageView;
    ImageView imageView2;
    Bitmap bmpSrc;
    Bitmap bmpFrame;
    Bitmap bmpDst;
    byte[] pictures;
    ExtractionTF preTF;
    int height;
    int width;
    private Button takePhotoBn;
    private Uri imageUri;
    String mCurrentPhotoPath;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        takePhotoBn = (Button) findViewById(R.id.takephoto);
        // Example of a call to a native method
        tv = (TextView) findViewById(R.id.sample_text);
        txt = (TextView) findViewById(R.id.txt_id);
        imageView = (ImageView) findViewById(R.id.imageView1);
        imageView2 = (ImageView) findViewById(R.id.imageView2);
        bmpSrc = BitmapFactory.decodeResource(getResources(), R.drawable.test_image);
        height=bmpSrc.getHeight();
        width=bmpSrc.getWidth();
        imageView.setImageBitmap(bmpSrc);
        pictures = new byte[256 * 256];
        preTF = new ExtractionTF(getAssets(), MODEL_FILE);//输入模型存放路径，并加载TensoFlow模型

        //检查相机权限
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED){
            ActivityCompat.requestPermissions(this,new String[]{Manifest.permission.CAMERA},1001);
        }
        initPhotoError();

        //检查存储权限
        verifyStoragePermissions(this);

        takePhotoBn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                //将File对象转换为Uri并启动照相程序
                //图片名称 时间命名
                //SimpleDateFormat format = new SimpleDateFormat("yyyyMMddHHmmss");
                //Date date = new Date(System.currentTimeMillis());
                //String filename = format.format(date);
                //创建File对象用于存储拍照的图片 SD卡根目录
                //File outputImage = new File(Environment.getExternalStorageDirectory(),"test.jpg");
                //存储至DCIM文件夹
                File path = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DCIM);
                File outputImage = new File(path,"dod.jpg");
                mCurrentPhotoPath = outputImage.getAbsolutePath();

                try {
                    if(outputImage.exists()) {
                        outputImage.delete();
                    }
                    outputImage.createNewFile();
                } catch(IOException e) {
                    e.printStackTrace();
                }
                //将File对象转换为Uri并启动照相程序
                imageUri = Uri.fromFile(outputImage);
                Intent intent = new Intent("android.media.action.IMAGE_CAPTURE"); //照相
                intent.putExtra(MediaStore.EXTRA_OUTPUT, imageUri); //指定图片输出地址
                startActivityForResult(intent,1); //启动照相

                //拍完照startActivityForResult() 结果返回onActivityResult()函数
            }
        });


    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == Activity.RESULT_OK) {
            bmpSrc = small(BitmapFactory.decodeFile(mCurrentPhotoPath));
            imageView.setImageBitmap(bmpSrc);

        }
    }

    public void click01(View v) {
        String res = "预测结果为：";
        float[] result = preTF.getPredict(bmpSrc);
        System.out.println(result.length);

        for (int i = 0; i < result.length; i++) {
            //Log.i(TAG, res+result[i] );
            //res=res+String.valueOf(result[i])+" ";
            if (result[i] > 0.0f)
                pictures[i] = (byte) 255;
            else
                pictures[i] = (byte) 0;
        }

        bmpFrame = createBitmap(pictures, 256, 256);
        imageView.setImageBitmap(bmpFrame);
        pictures = new byte[height * width];
        System.out.println(Environment.getExternalStorageDirectory().getPath());
        bmpDst = OpenCVUtils.getDistortion(bmpSrc,bmpFrame);
        if(bmpDst!=null){
            imageView2.setImageBitmap(bmpDst);
            Toast.makeText(MainActivity.this, "识别成功！",  Toast.LENGTH_LONG).show();
        }
        else{
            imageView2.setImageBitmap(null);
            Toast.makeText(MainActivity.this, "识别失败！",  Toast.LENGTH_LONG).show();
        }

    }

    /**
     * A native method that is implemented by the 'native-lib' native library,
     * which is packaged with this application.
     */

    public static Bitmap createBitmap(byte[] values, int picW, int picH) {
        if (values == null || picW <= 0 || picH <= 0)
            return null;
        //使用8位来保存图片
        Bitmap bitmap = Bitmap
                .createBitmap(picW, picH, Bitmap.Config.ARGB_8888);
        int pixels[] = new int[picW * picH];
        for (int i = 0; i < pixels.length; ++i) {
            //关键代码，生产灰度图
            pixels[i] = values[i] * 256 * 256 + values[i] * 256 + values[i] + 0xFF000000;
        }
        bitmap.setPixels(pixels, 0, picW, 0, 0, picW, picH);
        values = null;
        pixels = null;
        return bitmap;
    }

    public static void verifyStoragePermissions(Activity activity) {

        try {
            //检测是否有写的权限
            int permission = ActivityCompat.checkSelfPermission(activity,
                    "android.permission.WRITE_EXTERNAL_STORAGE");
            if (permission != PackageManager.PERMISSION_GRANTED) {
                // 没有写的权限，去申请写的权限，会弹出对话框
                ActivityCompat.requestPermissions(activity, PERMISSIONS_STORAGE,REQUEST_EXTERNAL_STORAGE);
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }



    private void initPhotoError(){
        // android 7.0系统解决拍照的问题
        StrictMode.VmPolicy.Builder builder = new StrictMode.VmPolicy.Builder();
        StrictMode.setVmPolicy(builder.build());
        builder.detectFileUriExposure();
    }

    private static Bitmap small(Bitmap bitmap) {
    Matrix matrix = new Matrix();
    matrix.postScale(0.5f,0.5f); //长和宽放大缩小的比例
    Bitmap resizeBmp = Bitmap.createBitmap(bitmap,0,0,bitmap.getWidth(),bitmap.getHeight(),matrix,true);
    return resizeBmp;
    }

}
