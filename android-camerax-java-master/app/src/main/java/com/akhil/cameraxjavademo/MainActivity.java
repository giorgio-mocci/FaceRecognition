package com.akhil.cameraxjavademo;

import androidx.annotation.NonNull;
import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.Camera;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageCapture;
import androidx.camera.core.ImageCaptureException;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.extensions.HdrImageCaptureExtender;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.lifecycle.LifecycleOwner;

import android.annotation.SuppressLint;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.media.Image;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.Looper;
import android.renderscript.ScriptGroup;
import android.util.Log;
import android.util.Size;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.akhil.cameraxjavademo.ml.Model;
import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.android.gms.tasks.Task;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil  ;


import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {

    private Executor executor = Executors.newSingleThreadExecutor();
    private int REQUEST_CODE_PERMISSIONS = 1001;
    private final String[] REQUIRED_PERMISSIONS = new String[]{"android.permission.CAMERA", "android.permission.WRITE_EXTERNAL_STORAGE"};


    private  Bitmap bitmapFOTO;
    PreviewView mPreviewView;
    TextView testo;
    ImageView immagine;
    OverlayView rettangolo;

    public Bitmap getResizedBitmap(Bitmap bm, int newWidth, int newHeight) {
        int width = bm.getWidth();
        int height = bm.getHeight();
        float scaleWidth = ((float) newWidth) / width;
        float scaleHeight = ((float) newHeight) / height;
        // CREATE A MATRIX FOR THE MANIPULATION
        Matrix matrix = new Matrix();
        // RESIZE THE BIT MAP
        matrix.postScale(scaleWidth, scaleHeight);

        // "RECREATE" THE NEW BITMAP
        Bitmap resizedBitmap = Bitmap.createBitmap(
                bm, 0, 0, width, height, matrix, false);
        bm.recycle();
        return resizedBitmap;
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {

        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        rettangolo = findViewById(R.id.overlayView);
        mPreviewView = findViewById(R.id.previewView);
        immagine = findViewById(R.id.imageView1);

        //this.setContentView(rettangolo);
        testo =(TextView)findViewById(R.id.textView7);

        if(allPermissionsGranted()){
            startCamera(); //start camera if permission has been granted by user
        } else{
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS);
        }
    }

    private void startCamera() {

        final ListenableFuture<ProcessCameraProvider> cameraProviderFuture = ProcessCameraProvider.getInstance(this);

        cameraProviderFuture.addListener(new Runnable() {
            @Override
            public void run() {
                try {

                    ProcessCameraProvider cameraProvider = cameraProviderFuture.get();
                    bindPreview(cameraProvider);



                } catch (ExecutionException | InterruptedException e) {
                    // No errors need to be handled for this Future.
                    // This should never be reached.
                }
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private Bitmap toBitmap(Image image) {
        Image.Plane[] planes = image.getPlanes();
        ByteBuffer yBuffer = planes[0].getBuffer();
        ByteBuffer uBuffer = planes[1].getBuffer();
        ByteBuffer vBuffer = planes[2].getBuffer();

        int ySize = yBuffer.remaining();
        int uSize = uBuffer.remaining();
        int vSize = vBuffer.remaining();

        byte[] nv21 = new byte[ySize + uSize + vSize];
        //U and V are swapped
        yBuffer.get(nv21, 0, ySize);
        vBuffer.get(nv21, ySize, vSize);
        uBuffer.get(nv21, ySize + vSize, uSize);

        YuvImage yuvImage = new YuvImage(nv21, ImageFormat.NV21, image.getWidth(), image.getHeight(), null);
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        yuvImage.compressToJpeg(new Rect(0, 0, yuvImage.getWidth(), yuvImage.getHeight()), 75, out);

        byte[] imageBytes = out.toByteArray();
        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
    }


    void bindPreview(@NonNull ProcessCameraProvider cameraProvider) {

        Preview preview = new Preview.Builder()
                .build();

        CameraSelector cameraSelector = new CameraSelector.Builder()
                .requireLensFacing(CameraSelector.LENS_FACING_BACK)
                .build();

        ImageAnalysis imageAnalysis = new ImageAnalysis.Builder()
          //      .setTargetResolution(new Size(mPreviewView.getWidth(),mPreviewView.getHeight()))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build();

        ImageCapture.Builder builder = new ImageCapture.Builder();

        //Vendor-Extensions (The CameraX extensions dependency in build.gradle)
        HdrImageCaptureExtender hdrImageCaptureExtender = HdrImageCaptureExtender.create(builder);

        // Query if extension is available (optional).
        if (hdrImageCaptureExtender.isExtensionAvailable(cameraSelector)) {
            // Enable the extension if available.
            hdrImageCaptureExtender.enableExtension(cameraSelector);
        }

        final ImageCapture imageCapture = builder
                .setTargetRotation(this.getWindowManager().getDefaultDisplay().getRotation())
                .build();

        preview.setSurfaceProvider(mPreviewView.createSurfaceProvider());
        imageAnalysis.setAnalyzer(executor, new ImageAnalysis.Analyzer() {
            @Override
            @SuppressLint("UnsafeExperimentalUsageError")
            public void analyze(@NonNull ImageProxy imageProxy) {


                int rotationDegrees = imageProxy.getImageInfo().getRotationDegrees();
                // insert your code here.
                FaceDetector detector = FaceDetection.getClient();
                InputImage image = InputImage.fromMediaImage(imageProxy.getImage(), rotationDegrees);
                bitmapFOTO = toBitmap(image.getMediaImage());




                float scale =   (float)imageProxy.getImage().getWidth()/ (float)imageProxy.getImage().getHeight() ;
                final  String stampare2 = "view camera res "+ new Size(mPreviewView.getWidth(),mPreviewView.getHeight())+ " image res "+ image.getWidth() +" " + image.getHeight();







                float xTranslate;
                float yTranslate;



                if (mPreviewView.getWidth() > mPreviewView.getHeight()) {
                   //  portrait: viewFinder width corresponds to target height
                    xTranslate = (mPreviewView.getWidth() - imageProxy.getImage().getHeight()) / 2f;
                    yTranslate = (mPreviewView.getHeight() - imageProxy.getImage().getWidth()) / 2f;
                } else {
                    // landscape: viewFinder width corresponds to target width
                    xTranslate = (mPreviewView.getWidth()- imageProxy.getImage().getWidth()) / 2f;
                    yTranslate = (mPreviewView.getHeight() - imageProxy.getImage().getHeight()) / 2f;
                }






                Task<List<Face>> result =
                        detector.process( InputImage.fromBitmap(bitmapFOTO, rotationDegrees) )
                                .addOnSuccessListener(
                                        new OnSuccessListener<List<Face>>() {
                                            @RequiresApi(api = Build.VERSION_CODES.Q)
                                            @Override
                                            public void onSuccess(List<Face> faces) {
                                                // Task completed successfully
                                                System.out.println("IN QUESTO MOMENTO CI SONO "+ faces.size()+ " FACCE RILEVATE");
                                                List<Rect> sources = new ArrayList<>();
                                                if (faces.size() >0)
                                                {
                                                    //testo.setText("HAI TROVATO UNA FACCIA!!");
                                                    String stampare  = "";
                                                    for (Face face : faces) {
                                                        Rect bounds = face.getBoundingBox();

                                                        if(bounds.top > 20) bounds.top -= 20;
                                                        if (bounds.left > 20) bounds.left -= 20;
                                                        bounds.right +=20;
                                                        bounds.bottom += 20;


                                                        stampare = "misureBitmap "+bitmapFOTO.getWidth() + " "+bitmapFOTO.getHeight() +" valori dopo "+
                                                                (bounds.right-bounds.left) + "  " + (bounds.bottom-bounds.top) + "   " + bounds.left + "  "+ bounds.top;
                                                        Matrix matrix = new Matrix();
                                                        matrix.postRotate(90);
                                                        bitmapFOTO = Bitmap.createBitmap(bitmapFOTO, 0, 0, bitmapFOTO.getWidth(), bitmapFOTO.getHeight(), matrix, true);
                                                        Bitmap croppedBitmap;
                                                        if(bounds.top >0 && bounds.left >0 && bounds.left + (bounds.right-bounds.left) <=bitmapFOTO.getWidth() &&
                                                                bounds.top +(bounds.bottom-bounds.top)<= bitmapFOTO.getHeight() )
                                                        {
                                                            croppedBitmap = Bitmap.createBitmap(bitmapFOTO, bounds.left, bounds.top, bounds.right-bounds.left , bounds.bottom-bounds.top);

                                                        }else croppedBitmap = bitmapFOTO;


                                                        //SCALA
                                                        bounds.top +=  ((bounds.top)*scale);
                                                        bounds.left += ((bounds.left)*scale);
                                                        bounds.right +=((bounds.right)*scale);
                                                        bounds.bottom += ((bounds.bottom)*scale);
                                                        bounds.top += (bounds.top/100) * 20;
                                                        bounds.bottom += (bounds.bottom/100) * 20;

                                                        int centerX =  ((bounds.top + bounds.bottom)/2);
                                                        int centerY = ((bounds.right + bounds.left)/2);



                                                        int lunghezzaRiga= mPreviewView.getWidth();
                                                        int altezzaRiga = mPreviewView.getHeight();







                                                        immagine.setImageBitmap(croppedBitmap);
                                                        sources.add(bounds);
                                                      // stampare += "top= "+bounds.top +" right= "+bounds.right +" bottom= "+bounds.bottom +   " left= "+bounds.left +"FATTORE SCALA = " + scale;

                                                        //testo.setText(stampare);


                                                        try {
                                                            MappedByteBuffer tfliteModel = FileUtil.loadMappedFile(getApplicationContext(), "model.tflite");
                                                            Interpreter tflite = new Interpreter(tfliteModel);

                                                            Bitmap resized =   Bitmap.createScaledBitmap(croppedBitmap,250,250,false);






                                                            int imageTensorIndex = 0;
                                                            DataType imageDataType = tflite.getInputTensor(imageTensorIndex).dataType();

                                                            TensorImage tfImage = new TensorImage(imageDataType);
                                                            tfImage.load(resized);

                                                            float [][] labelProbArray = new float[1][2];
                                                            tflite.run(tfImage.getBuffer(), labelProbArray);



                                                            if(labelProbArray != null) System.out.println("CACCAPUPU" +labelProbArray);
                                                               if(labelProbArray[0][0] >= 0.7) rettangolo.setTargets(sources,true);
                                                               else rettangolo.setTargets(sources,false);

                                                        //    testo.setText(" "+  tflite.getOutputTensor(0).shape()[0] +"   "+ tflite.getOutputTensor(0).shape()[1] + " " +    labelProbArray[0][0] + "  "+  labelProbArray[0][1]);
                                                        } catch (IOException e) {
                                                            // TODO Handle the exception
                                                        }


                                                    }



                                                   // sources.clear();


                                                    imageProxy.close();

                                                }else{

                                                    imageProxy.close();
                                                    rettangolo.setTargets(sources,false);
                                                    //testo.setText(stampare2);

                                                }
                                            }
                                        }

                                        )
                                .addOnFailureListener(
                                        new OnFailureListener() {
                                            @Override
                                            public void onFailure(@NonNull Exception e) {
                                                // Task failed with an exception
                                                // ...
                                            }
                                        });

                // after done, release the ImageProxy object

            }
        });
        Camera camera = cameraProvider.bindToLifecycle((LifecycleOwner)this, cameraSelector, preview, imageAnalysis, imageCapture);









    }

    public String getBatchDirectoryName() {

        String app_folder_path = "";
        app_folder_path = Environment.getExternalStorageDirectory().toString() + "/images";
        File dir = new File(app_folder_path);
        if (!dir.exists() && !dir.mkdirs()) {

        }

        return app_folder_path;
    }

    private boolean allPermissionsGranted(){

        for(String permission : REQUIRED_PERMISSIONS){
            if(ContextCompat.checkSelfPermission(this, permission) != PackageManager.PERMISSION_GRANTED){
                return false;
            }
        }
        return true;
    }
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {

        if(requestCode == REQUEST_CODE_PERMISSIONS){
            if(allPermissionsGranted()){
                startCamera();
            } else{
                Toast.makeText(this, "Permissions not granted by the user.", Toast.LENGTH_SHORT).show();
                this.finish();
            }
        }
    }
}
