package com.example.dolgio;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;

public class MainActivityNonChaquo extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main_non_chaquo);
    }


    public void live(View view) {


        Intent intent = new Intent(this, FiltraLive.class);
        startActivity(intent);

    }

    public void statico(View view) {


        Intent intent = new Intent(this, FiltraNonChaquo.class);
        startActivity(intent);

    }
}