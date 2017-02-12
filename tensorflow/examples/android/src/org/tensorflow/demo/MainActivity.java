package org.tensorflow.demo;

import android.Manifest;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.util.SparseArray;
import com.google.android.gms.vision.face.Face;
import com.google.android.gms.vision.face.Landmark;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.ArrayList;



/**
 * Created by gehen on 1/20/17.
 */

public class MainActivity extends AppCompatActivity {

    private FaceOverlayView mFaceOverlayView;
    private SparseArray<Face> mFaces;
    private Bitmap[] mCropFaces;
//    private int face_num;
    private static final int REQUEST_EXTERNAL_STORAGE = 1;
    private static String[] PERMISSIONS_STORAGE = {
            Manifest.permission.READ_EXTERNAL_STORAGE,
            Manifest.permission.WRITE_EXTERNAL_STORAGE
    };

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

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        mFaceOverlayView = (FaceOverlayView) findViewById( R.id.face_overlay );

        InputStream stream = getResources().openRawResource( R.raw.face );
        Bitmap bitmap = BitmapFactory.decodeStream(stream);

        mFaces = mFaceOverlayView.setBitmap(bitmap);
        mCropFaces = crop_face(bitmap, mFaces);
        ArrayList features = Extract_Features(mCropFaces);
        String[] person_list = {"person_a","person_b","person_c","person_d","person_e","person_f"};
        savetoFile(features,person_list);
        loadfromfile();

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
        double[] offset_pct = {0.2,0.2};
        int[] dest_sz = {100,130};
        int face_num = mFaces.size();
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

    public ArrayList Extract_Features(Bitmap[] cropFaces) {
        int face_num =cropFaces.length;
        init_classifier();
        ArrayList temp = new ArrayList();
        for (int i =0;i<face_num;i++) {
            temp.add(float2string(classifier.recognizeImage(cropFaces[i]).clone()));
        }
        return temp;
    }

    public StringBuilder float2string(float[] temp){
        StringBuilder a = new StringBuilder();
        for (int i=0;i<temp.length;i++){
            a.append(Float.toString(temp[i]));
            a.append(" ");
        }
        return a;
    }

    public float[] string2float(String temp){
        String[] nums = temp.split(" ");
        float[] values = new float[nums.length];
        for (int i=0;i<nums.length;i++){
            values[i] = Float.parseFloat(nums[i]);
        }
        return values;
    }

    private void writeToFile(String data) {
        try {
            OutputStreamWriter outputStreamWriter = new OutputStreamWriter(openFileOutput("features.txt", Context.MODE_PRIVATE));
            outputStreamWriter.write(data);
            outputStreamWriter.close();
        }
        catch (IOException e) {
            Log.e("Exception", "File write failed: " + e.toString());
        }
    }

    private String readFromFile() {

        String ret = "";

        try {
            InputStream inputStream = openFileInput("features.txt");

            if ( inputStream != null ) {
                InputStreamReader inputStreamReader = new InputStreamReader(inputStream);
                BufferedReader bufferedReader = new BufferedReader(inputStreamReader);
                String receiveString = "";
                StringBuilder stringBuilder = new StringBuilder();

                while ( (receiveString = bufferedReader.readLine()) != null ) {
                    stringBuilder.append(receiveString);
                }

                inputStream.close();
                ret = stringBuilder.toString();
            }
        }
        catch (FileNotFoundException e) {
            Log.e("login activity", "File not found: " + e.toString());
        } catch (IOException e) {
            Log.e("login activity", "Can not read file: " + e.toString());
        }

        return ret;
    }

    private String savetoFile(ArrayList features, String[] person){
        int num = person.length;
        String add="";
        for (int i=0;i<num;i++){
            add += person[i]+" "+features.get(i).toString()+" | ";
        }
        writeToFile(add);
        return add;
    }

    private ArrayList loadfromfile(){
        ArrayList temp = new ArrayList();
        String txt_data = readFromFile();
        int num = txt_data.length();
        int a = txt_data.indexOf("person_b");
        String[] pp = txt_data.split(" ");
        String[] aa = txt_data.split("\\s\\S\\s");
        return temp;
    }




}
