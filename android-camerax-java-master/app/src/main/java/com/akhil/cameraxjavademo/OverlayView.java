package com.akhil.cameraxjavademo;

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
            for (final Rect entry : this.targets) {
                System.out.println("RETTANGOLO");
                canvas.drawRect(entry, this.paint);
            }
        }
    }

    public void setTargets(@NonNull final List<Rect> sources,boolean Trump) {
        synchronized (this) {
            if (Trump)this.paint.setColor(Color.GREEN);
            if (!Trump)this.paint.setColor(Color.RED);
            this.targets.clear();
            this.targets.addAll(sources);
            this.postInvalidate();
        }
    }
}