package com.example.dolgio;

import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.PathUtils;
import androidx.loader.content.CursorLoader;

import android.content.Intent;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageSwitcher;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;
import android.widget.ViewSwitcher;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;
import java.util.StringTokenizer;

public class Output extends AppCompatActivity {
    ArrayList<Uri> mArrayUri;

    int position = 0;
    ImageSwitcher imageView;
    TextView tw;
    int total;




    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_output);
        this.mArrayUri = MainActivity.filtro.getUri();

        imageView = findViewById(R.id.image);
        imageView.setFactory(new ViewSwitcher.ViewFactory() {
            @Override
            public View makeView() {
                ImageView imageView1 = new ImageView(getApplicationContext());
                return imageView1;

            }
        });
        imageView.setImageURI(mArrayUri.get(0));
        tw = findViewById(R.id.textView5);
        total = mArrayUri.size();
        if(total==1)
            tw.setText("Ho trovato solo questa immagine");
        else
            tw.setText("immagine 1 di "+total);




    }

    public void previous(View view) {
        if (position > 0) {
            // decrease the position by 1
            position--;
            imageView.setImageURI(mArrayUri.get(position));
            updateTextView();

        }else Toast.makeText(Output.this, "First Image Already Shown", Toast.LENGTH_SHORT).show();


    }

    private void updateTextView(){
        int n = position +1;
        tw.setText("Immagine numero "+n+" di "+total);
    }


    public void next(View view) {
        if (position < mArrayUri.size() - 1) {
            // increase the position by 1
            position++;
            imageView.setImageURI(mArrayUri.get(position));
            updateTextView();


        } else {
            Toast.makeText(Output.this, "Last Image Already Shown", Toast.LENGTH_SHORT).show();
        }


    }



    @RequiresApi(api = Build.VERSION_CODES.O)
    public void salva(View view) {
        DateTimeFormatter dtf = DateTimeFormatter.ofPattern("yyyy_MM_dd_ss");
        LocalDateTime now = LocalDateTime.now();
        String title = dtf.format(now) + "" + this.position;



        //Bitmap bitmap = BitmapFactory.decodeFile(this.mArrayUri.get(this.position).toString());
        Bitmap bitmap = null;
        try {
            bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), this.mArrayUri.get(this.position));
        } catch (IOException e) {
            e.printStackTrace();
        }
        if (bitmap != null) {

            String savedImageURL = MediaStore.Images.Media.insertImage(
                    getContentResolver(),
                    bitmap,
                    title,
                    "Image"
            );


            tw.setText("immagine salvata in Pictures");
        } else {
            tw.setText("errore salvataggio foto");
        }


    }
}