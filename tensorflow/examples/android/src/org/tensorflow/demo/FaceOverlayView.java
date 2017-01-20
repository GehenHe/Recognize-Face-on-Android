package org.tensorflow.demo;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Rect;
import android.util.AttributeSet;
import android.util.Log;
import android.util.SparseArray;
import android.view.View;

import com.google.android.gms.vision.Frame;
import com.google.android.gms.vision.face.Face;
import com.google.android.gms.vision.face.FaceDetector;
import com.google.android.gms.vision.face.Landmark;

/**
 * Created by gehen on 1/18/17.
 */

public class FaceOverlayView extends View {
    private Bitmap mBitmap;
    private SparseArray<Face> mFaces;

    public FaceOverlayView(Context context) {
        this(context, null);
    }

    public FaceOverlayView(Context context, AttributeSet attrs) {
        this(context, attrs, 0);
    }

    public FaceOverlayView(Context context, AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
    }

    public SparseArray<Face> setBitmap(Bitmap bitmap){
        mBitmap = bitmap;
        FaceDetector detector = new FaceDetector.Builder( getContext() )
                .setTrackingEnabled(true)
                .setLandmarkType(FaceDetector.ALL_LANDMARKS)
                .setMode(FaceDetector.FAST_MODE)
                .build();
        if (!detector.isOperational()) {
            Log.e("detector","can not be reached");
            //Handle contingency
        } else {
            Frame frame = new Frame.Builder().setBitmap(bitmap).build();
            mFaces = detector.detect(frame);
            detector.release();
        }
        return mFaces;
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);

        if ((mBitmap != null) && (mFaces != null)) {
            double scale = drawBitmap(canvas);
            drawFaceBox(canvas, scale);
//            drawFaceLandmarks(canvas,scale);
            drawCropFaces(canvas,scale);
        }
    }

    private double drawBitmap(Canvas canvas) {
        double viewWidth = canvas.getWidth();
        double viewHeight = canvas.getHeight();
        double imageWidth = mBitmap.getWidth();
        double imageHeight = mBitmap.getHeight();
        double scale = Math.min(viewWidth / imageWidth, viewHeight / imageHeight);

        Rect destBounds = new Rect(0, 0, (int)(imageWidth * scale), (int)(imageHeight * scale));
        canvas.drawBitmap(mBitmap, null, destBounds, null);
        return scale;
    }

    private void drawCropFaces(Canvas canvas,double scale) {
        Bitmap[] cropfaces = crop_face(mBitmap,mFaces);
        double imageWidth = mBitmap.getWidth();
        double imageHeight = mBitmap.getHeight();
        double crop_width = cropfaces[1].getWidth();
        double crop_height = cropfaces[1].getHeight();
        int width = (int)(crop_height+40);
        int height = (int)(crop_height+40);
        int num = cropfaces.length;

        for(int i=0;i<num;i++){
            int x = (i%3*width);
            int y = (int)(i/3*height+imageHeight*scale);
            Rect destBounds = new Rect(x, y, (int)(crop_width * scale+x), (int)(crop_height * scale+y));
            canvas.drawBitmap(cropfaces[i],null,destBounds,null);
        }
    }

    private void drawFaceBox(Canvas canvas, double scale) {
        //This should be defined as a member variable rather than
        //being created on each onDraw request, but left here for
        //emphasis.
        Paint paint = new Paint();
        paint.setColor(Color.GREEN);
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(5);


        float left = 0;
        float top = 0;
        float right = 0;
        float bottom = 0;

        for( int i = 0; i < mFaces.size(); i++ ) {
            Face face = mFaces.valueAt(i);

            left = (float) ( face.getPosition().x * scale );
            top = (float) ( face.getPosition().y * scale );
            right = (float) scale * ( face.getPosition().x + face.getWidth() );
            bottom = (float) scale * ( face.getPosition().y + face.getHeight() );

            canvas.drawRect( left, top, right, bottom, paint );
        }
    }

    private void drawFaceLandmarks(Canvas canvas, double scale ) {
        Paint paint = new Paint();
        paint.setColor( Color.GREEN );
        paint.setStyle( Paint.Style.STROKE );
        paint.setStrokeWidth( 5 );
        int j=0;

        for( int i = 0; i < mFaces.size(); i++ ) {
            Face face = mFaces.valueAt(i);

            for ( Landmark landmark : face.getLandmarks() ) {
                int cx = (int) ( landmark.getPosition().x * scale );
                int cy = (int) ( landmark.getPosition().y * scale );
                if (j%8==0) canvas.drawCircle( cx, cy, 10, paint );
                if (j%8==1) canvas.drawCircle( cx, cy, 10, paint );
                j++;
            }

        }
    }

    private void logFaceData() {
        float smilingProbability;
        float leftEyeOpenProbability;
        float rightEyeOpenProbability;
        float eulerY;
        float eulerZ;
        for( int i = 0; i < mFaces.size(); i++ ) {
            Face face = mFaces.valueAt(i);

            smilingProbability = face.getIsSmilingProbability();
            leftEyeOpenProbability = face.getIsLeftEyeOpenProbability();
            rightEyeOpenProbability = face.getIsRightEyeOpenProbability();
            eulerY = face.getEulerY();
            eulerZ = face.getEulerZ();

            Log.e( "Tuts+ Face Detection", "Smiling: " + smilingProbability );
            Log.e( "Tuts+ Face Detection", "Left eye open: " + leftEyeOpenProbability );
            Log.e( "Tuts+ Face Detection", "Right eye open: " + rightEyeOpenProbability );
            Log.e( "Tuts+ Face Detection", "Euler Y: " + eulerY );
            Log.e( "Tuts+ Face Detection", "Euler Z: " + eulerZ );
        }
    }

    public Bitmap[] crop_face(Bitmap bitmap , SparseArray<Face> mFaces){
        int face_num = mFaces.size();
        double[] offset_pct = {0.2,0.2};
        int[] dest_sz = {100,130};
        double distance;
        double reference;
        double scale;
        int offset_h = (int)(offset_pct[0]*dest_sz[0]);
        int offset_v = (int)(offset_pct[1]*dest_sz[1]);
        if (face_num>0){
            Bitmap[] crop_faces = new Bitmap[face_num];
//            for (int i =0;i<face_num;i++){
//                Face face = mFaces.valueAt(i);
//                int left = (int) ( face.getPosition().x);
//                int top = (int) ( face.getPosition().y);
//                int right = (int) ( face.getPosition().x + face.getWidth() );
//                int bottom = (int)( face.getPosition().y + face.getHeight() );
//                crop_faces[i] = Bitmap.createBitmap(bitmap,left,top,right-left,bottom-top);
//            }
//            return crop_faces;
            for( int i = 0; i < mFaces.size(); i++ ) {
                Face face = mFaces.valueAt(i);
                int j = 0;
                double[] eyel = new double[2];
                double[] eyer = new double[2];
                for ( Landmark landmark : face.getLandmarks() ) {
                    int cx = (int) ( landmark.getPosition().x);
                    int cy = (int) ( landmark.getPosition().y);
                    if (j%8==0) {
                        eyel[0] = cx;
                        eyel[1] = cy;
                    }
                    if (j%8==1) {
                        eyer[0] = cx;
                        eyer[1] = cy;
                    }
                    j++;
                }
                distance = eyer[0]-eyel[0];
                reference = 0.5*dest_sz[0];
                scale = distance/reference;
                double[] crop_xy = {(eyel[0]+eyer[0])/2 - dest_sz[0]*scale/2, (eyel[1]+eyer[1])/2 - dest_sz[1]*scale/3};
                double[] crop_size = {dest_sz[0]*scale, dest_sz[1]*scale};
                Bitmap cropBitmap = Bitmap.createBitmap(bitmap,(int)crop_xy[0],(int)crop_xy[1],(int)crop_size[0],(int)crop_size[1]);
                Bitmap resizeBitmap = Bitmap.createScaledBitmap(cropBitmap,100,130,false);
                crop_faces[i] = resizeBitmap;
            }
            return crop_faces;
        }
        else{
            return null;
        }
    }

    public double Distance(double[] eyel,double[] eyer){
        double dx = eyel[0]-eyer[0];
        double dy = eyel[1]-eyer[1];
        double dist = Math.sqrt(dx*dx+dy*dy);
        return dist;
    }
}

