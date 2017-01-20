package org.tensorflow.demo;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.os.Bundle;
import android.os.SystemClock;
import android.support.v7.app.AppCompatActivity;
import android.util.SparseArray;
import android.view.View;

import com.google.android.gms.vision.face.Face;
import com.google.android.gms.vision.face.Landmark;

import org.tensorflow.demo.env.BorderedText;

import java.io.InputStream;
import java.util.List;

/**
 * Created by gehen on 1/20/17.
 */

public class MainActivity extends AppCompatActivity {

    private FaceOverlayView mFaceOverlayView;
    private SparseArray<Face> mFaces;
    public Bitmap[] mCropFaces;

    private static final int NUM_CLASSES = 1001;
    private static final int INPUT_SIZE = 224;
    private static final int IMAGE_MEAN = 117;
    private static final float IMAGE_STD = 1;
    private static final String INPUT_NAME = "input:0";
    private static final String OUTPUT_NAME = "output:0";

    private static final String MODEL_FILE = "file:///android_asset/tensorflow_inception_graph.pb";
    private static final String LABEL_FILE =
            "file:///android_asset/imagenet_comp_graph_label_strings.txt";

    private Classifier classifier;


    private long lastProcessingTimeMs;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        mFaceOverlayView = (FaceOverlayView) findViewById( R.id.face_overlay );

        InputStream stream = getResources().openRawResource( R.raw.face );
        Bitmap bitmap = BitmapFactory.decodeStream(stream);

        mFaces = mFaceOverlayView.setBitmap(bitmap);
        mCropFaces = crop_face(bitmap,mFaces);
        init_classifier();
        final long startTime = SystemClock.uptimeMillis();

        final float[] results = classifier.recognizeImage(mCropFaces[1]);
        lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

    }

    public void init_classifier(){
        try {
            classifier =
                    TensorFlowImageClassifier.create(
                            getAssets(),
                            MODEL_FILE,
                            LABEL_FILE,
                            NUM_CLASSES,
                            INPUT_SIZE,
                            IMAGE_MEAN,
                            IMAGE_STD,
                            INPUT_NAME,
                            OUTPUT_NAME);
        } catch (final Exception e) {
            throw new RuntimeException("Error initializing TensorFlow!", e);
        }
    }

    public Bitmap[] crop_face(Bitmap bitmap , SparseArray<Face> mFaces){
        int face_num = mFaces.size();
        double[] offset_pct = {0.2,0.2};
        int[] dest_sz = {100,130};
        double distance;
        double reference;
        double scale;
        if (face_num>0){
            Bitmap[] crop_faces = new Bitmap[face_num];
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
                Bitmap resizeBitmap = Bitmap.createScaledBitmap(cropBitmap,224,224,false);
                crop_faces[i] = resizeBitmap;
            }
            return crop_faces;
        }
        else{
            return null;
        }
    }
}
