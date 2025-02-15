package com.example.dolgio;

import android.content.Context;
import android.database.Cursor;
import android.net.Uri;
import android.os.Build;
import android.provider.MediaStore;
import org.json.JSONArray;
import org.json.JSONException;

import androidx.annotation.RequiresApi;
import androidx.loader.content.CursorLoader;

import com.chaquo.python.PyObject;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

public class Filtro {
    private ArrayList<Uri> mArrayUri;
    private Context context;
    private Map<String, Uri> map;

    public Filtro() {
        map = new HashMap<String, Uri>();


    }

    public void setContext(Context context) {
        this.context = context;
    }

    public ArrayList<Uri> getUri() {
        return this.mArrayUri;
    }

    public void setUri(ArrayList<Uri> a) {
        this.mArrayUri = (ArrayList<Uri>) a.clone();

    }

    @RequiresApi(api = Build.VERSION_CODES.O)
    public boolean prefiltra(){
        //pulisco la cartella
        File dir = new File("/data/user/0/com.example.dolgio/app_files");
        if(!dir.exists())
            dir.mkdir();
        for(File file: dir.listFiles())
            if (!file.isDirectory()) {
                System.out.println("elimino file "+file.getAbsolutePath()+" "+file.getName());
                file.delete();
            }
        System.out.println("filtra");
        Python py;


        //salvare tutte le foto in /data/data/com.example.dolgio/app_files
        if(!memorizza())
            return false;
        System.out.println("memorizzate");
        return true;
        //invocare detector.py
//        if (!Python.isStarted()) {
//            Python.start(new AndroidPlatform(context));
//        }
//        py = Python.getInstance();
//        return py;

    }


    @RequiresApi(api = Build.VERSION_CODES.O)
    public ArrayList<Uri> postfiltra(String json) {
        JSONArray str = null;
        try {
            str = new JSONArray(json);
        } catch (JSONException e) {
            System.out.println("lol");
            return null;
        }





        System.out.println("oggetto arrivato");
        if(str==null){
            System.out.println("errore");
            return null;
        }
//        System.out.println("stampo lista");
//        for(String s : str){
//            System.out.println(s);
//        }


        //generare la nuova lista e ritornarla
        ArrayList<Uri> result = new ArrayList<Uri>();
        System.out.println("genero nuova lista");
        for(int i =0; i<str.length(); i++){
            try {
                System.out.println(str.get(i)+"");
                result.add(this.map.get(str.get(i)+""));
            } catch (JSONException e) {
                System.out.println("errorejson");
            }

        }
        this.setUri(result);
        System.out.println("fine");
        int m[][] = new int[2][3];
        return result;




    }

    @RequiresApi(api = Build.VERSION_CODES.O)
    private boolean memorizza() {
        System.out.println("memorizza()");
        int position = 0;
        for (Uri uri : this.mArrayUri) {
            String filePath = this.getRealPathFromURI(uri);
            System.out.println(filePath);
            try {
                String file = position+".jpg";
                String nomeFile = "/data/user/0/com.example.dolgio/app_files/"+file;
                System.out.println(nomeFile);
                Files.copy(Paths.get(filePath), Paths.get(nomeFile), StandardCopyOption.REPLACE_EXISTING);
                this.map.put(file, uri);
            } catch (IOException e) {
                System.out.println("errore "+e.getMessage());
                System.out.println(e.toString()+" "+e.getLocalizedMessage());
                return false;
            }
            position++;

        }
        return true;
    }

    private String getRealPathFromURI(Uri contentUri) {
        String[] proj = {MediaStore.Images.Media.DATA};
        CursorLoader loader = new CursorLoader(context, contentUri, proj, null, null, null);
        Cursor cursor = loader.loadInBackground();
        int column_index = cursor.getColumnIndexOrThrow(MediaStore.Images.Media.DATA);
        cursor.moveToFirst();
        String result = cursor.getString(column_index);
        cursor.close();
        return result;
    }


}