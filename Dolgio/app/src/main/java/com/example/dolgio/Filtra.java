package com.example.dolgio;

import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;
import androidx.loader.content.CursorLoader;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.app.ProgressDialog;
import android.content.ClipData;
import android.content.Context;
import android.content.Intent;
import android.database.Cursor;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageSwitcher;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.Toast;
import android.widget.ViewSwitcher;

import com.chaquo.python.PyObject;
import com.chaquo.python.Python;

import org.json.JSONArray;
import org.json.JSONException;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;

public class Filtra extends AppCompatActivity {
    ArrayList<Uri> mArrayUri;
    int PICK_IMAGE_MULTIPLE = 1;
    int position = 0;
    ImageSwitcher imageView;
    Button ok;
    private static final String MSG_KEY = "yo";
    Context context;
    Python py;
    ProgressDialog progressDialog;


    @SuppressLint("HandlerLeak")
    private final Handler mHandler = new Handler() {
        @RequiresApi(api = Build.VERSION_CODES.O)
        @Override
        public void handleMessage(Message msg) {
            Bundle bundle = msg.getData();
            String string = bundle.getString(MSG_KEY);
            //dopo filtraggio
            ArrayList<Uri> tmp = MainActivity.filtro.postfiltra(string);
            Intent intent = new Intent(context, Output.class);
            progressDialog.dismiss();

            if (tmp == null) {
                Toast.makeText(context, "ERRORE", Toast.LENGTH_LONG).show();
                ok.setEnabled(true);
            } else if (tmp.isEmpty()) {
                Toast.makeText(context, "NESSUNA IMMAGINE COMBACIA", Toast.LENGTH_LONG).show();
                ok.setEnabled(true);
            } else
                context.startActivity(intent);
        }
    };


    private final Runnable mMessageSender = new Runnable() {
        public void run() {
            Message msg = mHandler.obtainMessage();
            Bundle bundle = new Bundle();

            PyObject obj = py.getModule("detector").callAttr("main");


            bundle.putString(MSG_KEY, obj.toString());
            msg.setData(bundle);
            mHandler.sendMessage(msg);
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_filtra);
        imageView = findViewById(R.id.image);
        mArrayUri = new ArrayList<Uri>();
        MainActivity.filtro.setContext(this);
        // showing all images in imageswitcher
        imageView.setFactory(new ViewSwitcher.ViewFactory() {
            @Override
            public View makeView() {
                ImageView imageView1 = new ImageView(getApplicationContext());
                return imageView1;
            }
        });
        ok = (Button) findViewById(R.id.okFiltra);
        ok.setEnabled(true);
        context = this;
        File dir = new File("/data/user/0/com.example.dolgio/app_files");
        if(!dir.exists())
            dir.mkdir();
        for(File file: dir.listFiles())
            if (!file.isDirectory()) {
                System.out.println("elimino file "+file.getAbsolutePath()+" "+file.getName());
                file.delete();
            }

        progressDialog = new ProgressDialog(Filtra.this);
        progressDialog.setMessage("Sto filtrando le tue foto..."); // Setting Message
        progressDialog.setTitle("Attendi"); // Setting Title
        progressDialog.setProgressStyle(ProgressDialog.STYLE_SPINNER); // Progress Dialog Style Spinner
        progressDialog.setCancelable(false);


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
            Toast.makeText(Filtra.this, "First Image Already Shown", Toast.LENGTH_SHORT).show();
        }


    }

    public void next(View view) {
        if (position < mArrayUri.size() - 1) {
            // increase the position by 1
            position++;
            imageView.setImageURI(mArrayUri.get(position));
        } else {
            Toast.makeText(Filtra.this, "Last Image Already Shown", Toast.LENGTH_SHORT).show();
        }


    }

    @RequiresApi(api = Build.VERSION_CODES.O)
    public void filtraggio(View view) {
        System.out.println(" FILTRAGGIO");

        progressDialog.show(); // Display Progress Dialog
        //Toast.makeText(Filtra.this, "Inizio a elaborare", Toast.LENGTH_SHORT).show();

        //MainActivity.filtro.setUri(mArrayUri);
        ok.setEnabled(false);
        System.out.println(" INIZIO PREFILTRA");
        boolean b =  MainActivity.filtro.prefiltra();
        System.out.println(" FINE PREFILTRA");
        py = MainActivity.py;
        if(py!=null && b)
            new Thread(mMessageSender).start();
        else{
            progressDialog.dismiss();
            ok.setEnabled(true);
            Toast.makeText(Filtra.this, "Errore: impossibile proseguire", Toast.LENGTH_SHORT).show();

        }
        System.out.println(" ECCO");


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
                MainActivity.filtro.setUri(mArrayUri);
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

    @Override
    protected void onPause() {
        super.onPause();
        //TODO not sure if this is needed for this use case
        mHandler.removeCallbacks(mMessageSender);
    }
}