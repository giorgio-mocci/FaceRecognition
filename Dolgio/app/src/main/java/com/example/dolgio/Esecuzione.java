package com.example.dolgio;

import android.content.Context;
import android.content.Intent;
import android.net.Uri;
import android.os.Build;
import android.widget.Toast;

import androidx.annotation.RequiresApi;

import java.util.ArrayList;

public class Esecuzione implements Runnable{


    private Context context;
    private ArrayList<Uri> mArrayUri;

    public Esecuzione(Context c, ArrayList<Uri> mArrayUri){
        this.context = c;
        this.mArrayUri = mArrayUri;
    }

    @RequiresApi(api = Build.VERSION_CODES.O)
    public void run(){
        System.out.println(" run");
        //LoadingDialog loadingDialog = new LoadingDialog(context);
       // loadingDialog.startDialog("Attendi mentre controllo le foto");
        Intent intent = new Intent(context, Output.class);
        MainActivity.filtro.setContext(context);
        MainActivity.filtro.setUri(this.mArrayUri);
        ArrayList<Uri> tmp = null;
        //loadingDialog.endDialog();
        //pd.dismiss();
        //loadingDialog.endDialog();
        if(tmp==null)
            Toast.makeText(context, "ERRORE", Toast.LENGTH_LONG).show();
        else if(tmp.isEmpty())
            Toast.makeText(context, "NESSUNA IMMAGINE COMBACIA", Toast.LENGTH_LONG).show();
        else
            context.startActivity(intent);
        System.out.println("  thread ended");

    }

}
