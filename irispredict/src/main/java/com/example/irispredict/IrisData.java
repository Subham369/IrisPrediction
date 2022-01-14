package com.example.irispredict;

import android.content.Context;
import android.widget.Toast;

import com.example.irispredict.ml.DfHeart;
import com.example.irispredict.ml.DfIris;
import com.example.irispredict.ml.Heart;
import com.example.irispredict.ml.Iris2;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.text.DecimalFormat;
import java.util.List;

public class IrisData {
    static String result="";
    public static String doInference(Context context,String input_value[]){


        int leng= input_value.length;
        float input_float[]=new float[leng];
        for(int i=0;i<leng;i++){
            input_float[i]= Float.parseFloat(input_value[i]);
        }


        ByteBuffer byteBuffer= ByteBuffer.allocateDirect(4*4);
        for(float i:input_float){
            byteBuffer.putFloat(i);
        }

        try {
            Iris2 model = Iris2.newInstance(context);

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 4}, DataType.FLOAT32);
            inputFeature0.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            Iris2.Outputs outputs = model.process(inputFeature0);
            //outputFeature0 = outputs.getOutputFeature0AsTensorBuffer().float;

            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();
            float[] output=outputFeature0.getFloatArray();//Converting outputFeature0 into float array

            DecimalFormat df =new DecimalFormat();
            df.setMaximumFractionDigits(3);
            result= "Iris Setosa : "+df.format(output[0])+"%\n"+"Iris Versicolor : "+df.format(output[1])+"%\n"+"Iris Virginica :"+
                    df.format(output[2])+"%";

            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
            Toast.makeText(context, e.getMessage(), Toast.LENGTH_SHORT).show();
        }

        return result;
    }

    public static String irisDecision(Context context,String input_value[]){


        int leng= input_value.length;
        float input_float[]=new float[leng];
        for(int i=0;i<leng;i++){
            input_float[i]= Float.parseFloat(input_value[i]);
        }


        ByteBuffer byteBuffer= ByteBuffer.allocateDirect(4*4);
        for(float i:input_float){
            byteBuffer.putFloat(i);
        }

        try {
            DfIris model = DfIris.newInstance(context);

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 4}, DataType.FLOAT32);
            inputFeature0.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            DfIris.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();
            float[] output=outputFeature0.getFloatArray();//Converting outputFeature0 into float array

            DecimalFormat df =new DecimalFormat();
            df.setMaximumFractionDigits(3);
            result= "Iris Setosa : "+df.format(output[0])+"%\n"+"Iris Versicolor : "+df.format(output[1])+"%\n"+"Iris Virginica :"+
                    df.format(output[2])+"%";

            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
            Toast.makeText(context, e.getMessage(), Toast.LENGTH_SHORT).show();
        }

        return result;
    }


    public static String heartPred(Context context, String[] input_value){

        int leng= input_value.length;
        float input_float[]=new float[leng];
        for(int i=0;i<leng;i++){
            input_float[i]= Float.parseFloat(input_value[i]);
        }


        ByteBuffer byteBuffer= ByteBuffer.allocateDirect(13*4);
        for(float i:input_float){
            byteBuffer.putFloat(i);
        }

        try {
            Heart model = Heart.newInstance(context);

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 13}, DataType.FLOAT32);
            inputFeature0.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            Heart.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();
            float[] output=outputFeature0.getFloatArray();//Converting outputFeature0 into float array
            if(output[0]==0){
                result="No Heart Disease";
            }
            else{
                result="Heart Disease";
            }
            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
            Toast.makeText(context, e.getMessage(), Toast.LENGTH_SHORT).show();
        }

        return result;
    }


    public static String heartDecision(Context context, String[] input_value){

        int leng= input_value.length;
        float input_float[]=new float[leng];
        for(int i=0;i<leng;i++){
            input_float[i]= Float.parseFloat(input_value[i]);
        }


        ByteBuffer byteBuffer= ByteBuffer.allocateDirect(13*4);
        for(float i:input_float){
            byteBuffer.putFloat(i);
        }

        try {
            DfHeart model = DfHeart.newInstance(context);

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 13}, DataType.FLOAT32);
            inputFeature0.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            DfHeart.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();
            float[] output=outputFeature0.getFloatArray();//Converting outputFeature0 into float array
            if(output[0]==0){
                result="No Heart Disease";
            }
            else{
                result="Heart Disease";
            }
            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
            Toast.makeText(context, e.getMessage(), Toast.LENGTH_SHORT).show();
        }

        return result;
    }

}
