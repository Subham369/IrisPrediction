package com.example.irisprediction;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;

import com.example.irispredict.IrisData;

public class MainActivity extends AppCompatActivity {

    private EditText ed1,ed2,ed3,ed4;
    private TextView txt_output;
    private Button predict,reset;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        ed1=findViewById(R.id.edt1Input);
        ed2=findViewById(R.id.edt2Input);
        ed3=findViewById(R.id.edt3Input);
        ed4=findViewById(R.id.edt4Input);
        txt_output=findViewById(R.id.txt_result);
        predict=findViewById(R.id.btn_predict);
        reset=findViewById(R.id.btn_reset);

        reset.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                ed1.getText().clear();
                ed2.getText().clear();
                ed3.getText().clear();
                ed4.getText().clear();
                ed4.clearFocus();
                txt_output.setText("");
            }
        });


        predict.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if(ed1.getText().length()!=0&&ed2.getText().length()!=0&&ed3.getText().length()!=0&&ed4.getText().length()!=0) {
                    String v1 = ed1.getText().toString();
                    String v2 = ed2.getText().toString();
                    String v3 = ed3.getText().toString();
                    String v4 = ed4.getText().toString();

                    String input_value[] =new String[4];
                    input_value[0]=v1;
                    input_value[1]=v2;
                    input_value[2]=v3;
                    input_value[3]=v4;

                    String result= IrisData.doInference(MainActivity.this,input_value);
                    txt_output.setText(String.valueOf(result));
                }
            }
        });
    }
}