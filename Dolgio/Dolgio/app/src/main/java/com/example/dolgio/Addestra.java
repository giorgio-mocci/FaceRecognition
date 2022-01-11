package com.example.dolgio;

import androidx.appcompat.app.AppCompatActivity;

import android.content.ClipData;
import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.view.View;
import android.widget.ImageSwitcher;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;
import android.widget.ViewSwitcher;

import java.util.ArrayList;

public class Addestra extends AppCompatActivity {
    ArrayList<Uri> mArrayUri;
    int PICK_IMAGE_MULTIPLE = 1;
    int position = 0;
    ImageSwitcher imageView;
    TextView tw;
    int total;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_addestra);

        imageView = findViewById(R.id.image);
        mArrayUri = new ArrayList<Uri>();
        //MainActivity.filtro.setContext(this);
        // showing all images in imageswitcher
        imageView.setFactory(new ViewSwitcher.ViewFactory() {
            @Override
            public View makeView() {
                ImageView imageView1 = new ImageView(getApplicationContext());
                return imageView1;
            }
        });
    }

    public void seleziona(View view) {


        Intent intent = new Intent();

        // setting type to select to be image
        intent.setType("image/*");

        // allowing multiple image to be selected
        intent.putExtra(Intent.EXTRA_ALLOW_MULTIPLE, true);
        intent.setAction(Intent.ACTION_GET_CONTENT);
        startActivityForResult(Intent.createChooser(intent, "Select Picture"), PICK_IMAGE_MULTIPLE);


    }

    public void addestramento(View view){
        Toast.makeText(Addestra.this, "Addestramento completato!", Toast.LENGTH_SHORT).show();
        System.out.println("addestramento...");
        Intent intent = new Intent(this, Filtra.class);
        startActivity(intent);
    }

    public void previous(View view) {
        if (position > 0) {
            // decrease the position by 1
            position--;
            imageView.setImageURI(mArrayUri.get(position));


        } else {
            Toast.makeText(Addestra.this, "First Image Already Shown", Toast.LENGTH_SHORT).show();
        }


    }

    public void next(View view) {
        if (position < mArrayUri.size() - 1) {
            // increase the position by 1
            position++;
            imageView.setImageURI(mArrayUri.get(position));
        } else {
            Toast.makeText(Addestra.this, "Last Image Already Shown", Toast.LENGTH_SHORT).show();
        }


    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {


        super.onActivityResult(requestCode, resultCode, data);
        // When an Image is picked
        if (requestCode == PICK_IMAGE_MULTIPLE && resultCode == RESULT_OK && null != data) {
            // Get the Image from data
            if (data.getClipData() != null) {
                ClipData mClipData = data.getClipData();
                int cout = data.getClipData().getItemCount();


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