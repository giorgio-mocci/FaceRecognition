package com.example.dolgio;

import androidx.annotation.NonNull;
import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;

import android.app.ProgressDialog;
import android.content.ClipData;
import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.graphics.Rect;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageSwitcher;
import android.widget.ImageView;
import android.widget.Toast;
import android.widget.ViewSwitcher;

import com.chaquo.python.Python;
import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.android.gms.tasks.Task;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.image.TensorImage;

import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.util.ArrayList;
import java.util.List;

public class FiltraNonChaquo extends AppCompatActivity {
    ArrayList<Uri> mArrayUri;
    ArrayList<Uri> result;
    int PICK_IMAGE_MULTIPLE = 1;
    int position = 0;
    ImageSwitcher imageView;
    Button ok;
    private static final String MSG_KEY = "yo";
    Context context;
    FaceDetector detector;

    ProgressDialog progressDialog;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_filtra_non_chaquo);
        imageView = findViewById(R.id.imageNC);
        mArrayUri = new ArrayList<Uri>();
        result = new ArrayList<Uri>();
        imageView.setFactory(new ViewSwitcher.ViewFactory() {
            @Override
            public View makeView() {
                ImageView imageView1 = new ImageView(getApplicationContext());
                return imageView1;
            }
        });
        ok = (Button) findViewById(R.id.okFiltraNC);
        ok.setEnabled(true);
        context = this;
        progressDialog = new ProgressDialog(FiltraNonChaquo.this);
        progressDialog.setMessage("Sto filtrando le tue foto..."); // Setting Message
        progressDialog.setTitle("Attendi"); // Setting Title
        progressDialog.setProgressStyle(ProgressDialog.STYLE_SPINNER); // Progress Dialog Style Spinner
        progressDialog.setCancelable(false);
        detector = FaceDetection.getClient();
    }

    public void seleziona(View view) {
        ok.setEnabled(true);
        Intent intent = new Intent();

        // setting type to select to be image
        intent.setType("image/*");

        // allowing multiple image to be selected
        intent.putExtra(Intent.EXTRA_ALLOW_MULTIPLE, true);
        intent.setAction(Intent.ACTION_GET_CONTENT);
        startActivityForResult(Intent.createChooser(intent, "Select Picture"), PICK_IMAGE_MULTIPLE);


    }

    public void previous(View view) {
        if (position > 0) {
            // decrease the position by 1
            position--;
            imageView.setImageURI(mArrayUri.get(position));


        } else {
            Toast.makeText(FiltraNonChaquo.this, "First Image Already Shown", Toast.LENGTH_SHORT).show();
        }


    }

    public void next(View view) {
        if (position < mArrayUri.size() - 1) {
            // increase the position by 1
            position++;
            imageView.setImageURI(mArrayUri.get(position));
        } else {
            Toast.makeText(FiltraNonChaquo.this, "Last Image Already Shown", Toast.LENGTH_SHORT).show();
        }


    }

    public void filtraggio(View view) {
        boolean found = false;
        System.out.println(" FILTRAGGIO");

        progressDialog.show(); // Display Progress Dialog
        ok.setEnabled(false);


        try {
            //caricamento del modello tflite
            MappedByteBuffer tfliteModel = FileUtil.loadMappedFile(getApplicationContext(), "model.tflite");
            //inizializzazione dell'interprete
            Interpreter tflite = new Interpreter(tfliteModel);


            Bitmap bitmapFOTO;
            List<Face> facce;
            Bitmap croppedBitmap, resized;
            System.out.println(" eccoci");


            for (Uri uri : this.mArrayUri) {
                System.out.println(" for");
                found = false; // fino a prova contraria non ho trovato nulla


                bitmapFOTO = MediaStore.Images.Media.getBitmap(this.getContentResolver(), uri); //da uri a bitmap
                System.out.println(bitmapFOTO+ " ciao");
                Task<List<Face>> resultFace;

                //per ogni foto selezionata si ricavano tutte le facce presenti in essa
                resultFace = detector.process(InputImage.fromBitmap(bitmapFOTO, 0));


                while(!resultFace.isComplete()){
                    //AHAHAH ATTESA ATTIVA SPERIAMO NESSUNO GUARDI MAI QUESTO CODICE
                }


                System.out.println(" resultFace");
                System.out.println(resultFace+" ciao");
                facce = resultFace.getResult();
                System.out.println(" faccie");
                System.out.println(facce+ " ciao");
                for (Face faccia : facce) {
                    Rect bounds = faccia.getBoundingBox();
                    System.out.println(" ciclo facce");
                    if (bounds.top > 20) bounds.top -= 20;
                    if (bounds.left > 20) bounds.left -= 20;
                    bounds.right += 20;
                    bounds.bottom += 20;


                    Matrix matrix = new Matrix();
                    matrix.postRotate(90);
                    bitmapFOTO = Bitmap.createBitmap(bitmapFOTO, 0, 0, bitmapFOTO.getWidth(), bitmapFOTO.getHeight(), matrix, true);
                    System.out.println(bitmapFOTO+" ciao");
                    if (bounds.top > 0 && bounds.left > 0 && bounds.left + (bounds.right - bounds.left) <= bitmapFOTO.getWidth() &&
                            bounds.top + (bounds.bottom - bounds.top) <= bitmapFOTO.getHeight()) {
                        croppedBitmap = Bitmap.createBitmap(bitmapFOTO, bounds.left, bounds.top, bounds.right - bounds.left, bounds.bottom - bounds.top);

                    } else croppedBitmap = bitmapFOTO;
                    System.out.println(" sto per resize");

                    resized = Bitmap.createScaledBitmap(croppedBitmap, 250, 250, false);
                    int imageTensorIndex = 0;
                    DataType imageDataType = tflite.getInputTensor(imageTensorIndex).dataType();
                    TensorImage tfImage = new TensorImage(imageDataType);
                    tfImage.load(resized);
                    float[][] labelProbArray = new float[1][2];
                    tflite.run(tfImage.getBuffer(), labelProbArray);



                    if (labelProbArray != null)
                        System.out.println("  result " + labelProbArray);
                    if (labelProbArray[0][0] >= 0.7)
                        found = true;
                    else
                        found = false;
                    System.out.println(found);


                }
                if(found)
                    this.result.add(uri);
            }


        } catch (Exception e) {
            e.printStackTrace();
            System.out.println(" ERRORE");
            // TODO Handle the exception
            progressDialog.dismiss();
            ok.setEnabled(true);
            Toast.makeText(FiltraNonChaquo.this, "Errore: impossibile proseguire", Toast.LENGTH_SHORT).show();
            System.out.println(e.getLocalizedMessage()+ " "+e.getMessage());
            System.out.println(e.toString());
        }

        progressDialog.dismiss();
        ok.setEnabled(true);
        if(result.size()==0)
            Toast.makeText(FiltraNonChaquo.this, "Nessuna faccia combacia", Toast.LENGTH_SHORT).show();
        else{
            MainActivity.filtro.setUri(this.result);
            Intent intent = new Intent(this, OutputNonChaquo.class);
            startActivity(intent);
        }

        //chiamo output non chaquo

    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        ok.setEnabled(true);
        super.onActivityResult(requestCode, resultCode, data);
        // When an Image is picked
        if (requestCode == PICK_IMAGE_MULTIPLE && resultCode == RESULT_OK && null != data) {
            // Get the Image from data
            if (data.getClipData() != null) {
                ClipData mClipData = data.getClipData();
                int cout = data.getClipData().getItemCount();
                mArrayUri.clear();
                for (int i = 0; i < cout; i++) {
                    // adding imageuri in array
                    Uri imageurl = data.getClipData().getItemAt(i).getUri();
                    mArrayUri.add(imageurl);
                }
                // setting 1st selected image into image switcher
                imageView.setImageURI(mArrayUri.get(0));
                position = 0;


            } else {
                Uri imageurl = data.getData();
                mArrayUri.add(imageurl);
                imageView.setImageURI(mArrayUri.get(0));
                position = 0;
            }
        } else {
            // show this if no image is selected
            Toast.makeText(this, "You haven't picked Image", Toast.LENGTH_LONG).show();
        }
    }


}