package com.example.dolgio;

import android.app.Activity;
import android.app.AlertDialog;
import android.app.Dialog;
import android.content.Context;
import android.graphics.Color;
import android.graphics.drawable.ColorDrawable;
import android.view.LayoutInflater;
import android.widget.TextView;


public class LoadingDialog {

    private Context context;
    private Dialog alertDialog;

    public LoadingDialog(Context context){
        this.context=context;
    }

    void startDialog(String title){
        alertDialog = new Dialog(context);
        alertDialog.setContentView(R.layout.dialog);
        alertDialog.getWindow().setBackgroundDrawable(new ColorDrawable(Color.TRANSPARENT));
        TextView tw =alertDialog.findViewById(R.id.textView7);
        tw.setText(title);

        alertDialog.create();
        alertDialog.show();

    }

    void endDialog(){
        alertDialog.dismiss();

    }
}
