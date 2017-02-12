package org.tensorflow.demo;

import android.Manifest;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.util.SparseArray;

import com.google.android.gms.drive.query.internal.MatchAllFilter;
import com.google.android.gms.vision.face.Face;
import com.google.android.gms.vision.face.Landmark;

import junit.framework.Test;

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
    private ArrayList ID_features = new ArrayList();
    private ArrayList ID_persons = new ArrayList();
    private ArrayList Test_features = new ArrayList();
    private ArrayList Test_Persons = new ArrayList();
    private ArrayList Test_Scores = new ArrayList();
    private boolean SAVE_FLAG=false;
    private boolean LOAD_FLAG=true;
    private double thresh = 0.9;

//    };

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
        int num = mFaces.size();
        if (num>0){
            mCropFaces = crop_face(bitmap, mFaces);
            Test_features = Extract_Features(mCropFaces);
//            Test_Persons.add("person_a");
//            Test_Persons.add("person_b");
//            Test_Persons.add("person_c");
//            Test_Persons.add("person_d");
//            Test_Persons.add("person_e");
//            Test_Persons.add("person_f");
        }
//        if (SAVE_FLAG==true)
//            SavetoFile(Test_features,Test_Persons);
        if (LOAD_FLAG==true)
            Loadfromfile();

        CompareFeature();
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

    private String SavetoFile(ArrayList features, ArrayList person){
        int num = person.size();
        String add="";
        for (int i=0;i<num;i++){
            add += person.get(i).toString()+" "+features.get(i).toString()+" | ";
        }
        writeToFile(add);
        return add;
    }

    public void Loadfromfile(){
        ID_persons.clear();
        ID_features.clear();
        ArrayList temp = new ArrayList();
        String txt_data = readFromFile();
        String[] features = txt_data.split("\\s\\S\\s");
        int num = features.length;
        for (int i=0;i<num;i++){
            temp.add(features[i]);
            String[] temp_split = features[i].split(" ",2);
            ID_persons.add(temp_split[0]);
            ID_features.add(temp_split[1]);
        }
    }

    public void CompareFeature(){
//        Test_features.clear();
//        Test_Scores.clear();
//        Test_Persons.clear();
        int test_num = Test_features.size();
        float[][] test_features = new float[test_num][];
        for (int i=0;i<test_num;i++){
            test_features[i] = string2float(Test_features.get(i).toString());
        }
        int id_num = ID_features.size();
        float[][] id_features = new float[id_num][];
        for (int i=0;i<id_num;i++){
            id_features[i] = string2float(ID_features.get(i).toString());
        }
        for (int i=0;i<test_num;i++){
            double[] score = new double[id_num];
            for (int j=0;j<id_num;j++){
                score[j] = cos_dist(test_features[i],id_features[j]);
            }
            int index = max_index(score);
            if (score[index]>thresh){
                Test_Persons.add(ID_persons.get(index).toString());
                Test_Scores.add(Float.toString((float)score[index]));
            }
            else
                Test_Persons.add("unknown");
        }
    }



    public int max_index(double[] data){
        double max = 0;
        int index = 0;
        int num = data.length;
        for(int i=0;i<num;i++){
            if(data[i]>max) {
                max = data[i];
                index = i;
            }
        }
        return index;
    }

    public float E_dist(float[] a,float[] b){
        double sum = 0;
        int num = a.length;
        for (int i=0;i<num;i++){
            sum+=(a[i]-b[i])*(a[i]-b[i]);
        }
        return (float)Math.sqrt(sum);
    }

    public double cos_dist(float[] a,float[] b){
        double dist = 0;
        int num = a.length;
        double sum = 0;
        for (int i=0;i<num;i++){
            sum+=a[i]*b[i];
        }
        dist = Math.abs(sum)/(norm(a)*norm(b));
        return dist;
    }

    public double norm(float[] a){
        int num = a.length;
        double sum = 0;
        for (int i=0;i<num;i++){
            sum+=a[i]*a[i];
        }
        return Math.sqrt(sum);
    }




}
