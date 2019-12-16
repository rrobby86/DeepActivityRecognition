package it.unibo.disi.deepactivityrecognition;

import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.widget.TextView;

public class MainActivity extends AppCompatActivity {

    private static final String[] ACTIVITIES = {
            "Walking",
            "Walking Upstairs",
            "Walking Downstairs",
            "Sitting",
            "Standing",
            "Laying"
    };

    private ActivityClassifier mClassifier;
    private TextView mText;
    private TextView mSubtext;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        mText = findViewById(R.id.text);
        mSubtext = findViewById(R.id.subtext);
        mClassifier = new ActivityClassifier();
        ActivityClassifier.Callback classifierCallback =
                new ActivityClassifier.Callback() {
            @Override
            public void activityInferred(int activity) {
                mText.setText(ACTIVITIES[activity]);
            }
            @Override
            public void bufferUpdated(int filled, int total, float[] values) {
                //mSubtext.setText(filled + " / " + total);
                if (values != null) {
                    mSubtext.setText(String.format("%6.3f %6.3f %6.3f", values[0], values[1], values[2]));
                } else {
                    mSubtext.setText("");
                }
            }
        };
        mClassifier.init(this, classifierCallback);
    }

    @Override
    protected void onPause() {
        super.onPause();
        mClassifier.stop();
    }

    @Override
    protected void onResume() {
        super.onResume();
        mClassifier.start();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        mClassifier.close();
    }

}
