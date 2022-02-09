package com.example.dolgio;

import android.Manifest;

import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.Bundle;
import android.view.View;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.chaquo.python.PyObject;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;


import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;

import java.io.InputStream;
import java.io.OutputStream;


public class MainActivity extends AppCompatActivity {

    private static final int EXTERNAL_STORAGE = 0;
    private static final int CAMERA = 1;
    static Python py;




    public static Filtro filtro;


    TextView tw;

    public void checkPermission(String permission, int requestCode) {
        if (ContextCompat.checkSelfPermission(MainActivity.this, permission) == PackageManager.PERMISSION_DENIED) {

            //
            // Requesting the permission
            ActivityCompat.requestPermissions(MainActivity.this, new String[]{permission}, requestCode);
        } else {
            //Toast.makeText(MainActivity.this, "Permission already granted", Toast.LENGTH_SHORT).show();
        }
    }



    //@RequiresApi(api = Build.VERSION_CODES.O)
    @RequiresApi(api = Build.VERSION_CODES.O)
    @Override
    protected void onCreate(Bundle savedInstanceState) {

        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        checkPermission(Manifest.permission.WRITE_EXTERNAL_STORAGE, EXTERNAL_STORAGE);
        checkPermission(Manifest.permission.CAMERA, CAMERA);

        checkPermission(Manifest.permission.READ_EXTERNAL_STORAGE, EXTERNAL_STORAGE);


        tw = findViewById(R.id.textView4);




        filtro = new Filtro();
        if (!Python.isStarted()) {
            Python.start(new AndroidPlatform(MainActivity.this));
        }
        py = Python.getInstance();


//        if (!Python.isStarted()) {
//            Python.start(new AndroidPlatform(this));
//        }
//        py = Python.getInstance();
//        pyo = py.getModule("detector");
//        obj = pyo.callAttr("main");
//
//
//        if (obj != null)
//            tw.setText(obj.toString());
//        else
//            tw.setText("boh");


    }

    public void chaquo(View view) {


        Intent intent = new Intent(this, Filtra.class);
        startActivity(intent);

    }

    public void nonChaquo(View view) {


        Intent intent = new Intent(this, MainActivityNonChaquo.class);
        startActivity(intent);

    }

    @Override
    public void onRequestPermissionsResult(int requestCode,
                                           @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode,
                permissions,
                grantResults);
        if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            Toast.makeText(MainActivity.this, "Granted " + requestCode, Toast.LENGTH_SHORT).show();
        } else {
            Toast.makeText(MainActivity.this, "Denied " + requestCode, Toast.LENGTH_SHORT).show();
        }
//        if (requestCode == CAMERA_PERMISSION_CODE) {
//            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
//                Toast.makeText(MainActivity.this, "Camera Permission Granted", Toast.LENGTH_SHORT) .show();
//            }
//            else {
//                Toast.makeText(MainActivity.this, "Camera Permission Denied", Toast.LENGTH_SHORT) .show();
//            }
//        }
//        else if (requestCode == STORAGE_PERMISSION_CODE) {
//            if (grantResults.length > 0
//                    && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
//                Toast.makeText(MainActivity.this, "Storage Permission Granted", Toast.LENGTH_SHORT).show();
//            } else {
//                Toast.makeText(MainActivity.this, "Storage Permission Denied", Toast.LENGTH_SHORT).show();
//            }
//        }
    }
}