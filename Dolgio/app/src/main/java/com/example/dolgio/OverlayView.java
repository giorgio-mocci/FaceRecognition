package com.example.dolgio;




import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Rect;
import android.util.AttributeSet;
import android.view.View;
import android.widget.RelativeLayout;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import java.util.ArrayList;
import java.util.List;



public final class OverlayView extends View {
    private final Paint paint = new Paint();
    private final List<Rect> targets = new ArrayList<>();
    private  ArrayList<Boolean> colori = new ArrayList<Boolean>();
    public OverlayView(@NonNull final Context context) {
        this(context, null);
    }

    public OverlayView(@NonNull final Context context, @Nullable final AttributeSet attrs) {
        this(context, attrs, 0);
    }

    public OverlayView(@NonNull final Context context, @Nullable final AttributeSet attrs, final int defStyleAttr) {
        super(context, attrs, defStyleAttr);

        final float density = context.getResources().getDisplayMetrics().density;
        this.paint.setStrokeWidth(2.0f * density);
        this.paint.setColor(Color.BLUE);
        this.paint.setStyle(Paint.Style.STROKE);
    }

    @Override
    protected void onDraw(final Canvas canvas) {
        super.onDraw(canvas);
        System.out.println("SONO DENTRO");
        synchronized (this) {

            for(int i=0; i<this.targets.size();i++)
            {
                if(colori.get(i))
                {
                    this.paint.setColor(Color.GREEN);
                    canvas.drawRect(this.targets.get(i), this.paint);
                }
                else{
                    this.paint.setColor(Color.RED);
                    canvas.drawRect(this.targets.get(i), this.paint);
                }
            }


        }
    }

    public void setTargets(@NonNull final List<Rect> sources,ArrayList<Boolean> Trump) {
        synchronized (this) {
            System.out.println("sono qua dentro" + Trump);
            if(sources.size()>0)
            System.out.println("rettangoli" + sources.get(0));

            colori = Trump;



            this.targets.clear();
            this.targets.addAll(sources);
            this.postInvalidate();
        }
    }
}