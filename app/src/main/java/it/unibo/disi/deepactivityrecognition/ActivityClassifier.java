package it.unibo.disi.deepactivityrecognition;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.util.Log;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.HashMap;
import java.util.Map;

/**
 * Componente per il riconoscimento dell'attività dai sensori.
 */
public class ActivityClassifier {

    private static final boolean ACCELEROMETER_ONLY = false;

    public interface Callback {
        void bufferUpdated(int filled, int total, float[] values);
        void activityInferred(int activity);
    }

    private static final String TAG = "ActivityClassifier";

    private static final String MODEL_ASSET = ACCELEROMETER_ONLY ?
            "activity_recognition.acc.tflite" : "activity_recognition.tflite";

    private static final int NUM_TIMESTEPS = 128;
    private static final int NUM_SENSORS = ACCELEROMETER_ONLY ? 1 : 3;
    private static final int NUM_AXES = 3;
    private static final int NUM_CLASSES = 6;

    private Callback callback;
    private Interpreter model = null;
    private SensorManager sensorManager;
    private SensorEventListener sensorListener;
    private Map<Integer, Integer> sensors;
    private float[][] inputBuffer;
    private float[][] outputBuffer;
    private int inputFilledMask;
    private int inputFillStep;

    /**
     * Inizializza il classificatore.
     * @param context contesto dell'applicazione
     * @param callback oggetto a cui mandare notifiche
     */
    public void init(Context context, Callback callback) {
        this.callback = callback;
        // carica modello TF Lite
        Log.d(TAG, "loading model");
        try {
            model = new Interpreter(loadModelFile(context));
        } catch (IOException e) {
            Log.e(TAG, "error loading model", e);
            if (callback != null) {
                callback.activityInferred(-1);
            }
        }
        Log.d(TAG, "model loaded");
        // inizializza e configura sensori
        sensors = new HashMap<>();
        sensors.put(Sensor.TYPE_ACCELEROMETER, 0);
        if (!ACCELEROMETER_ONLY) {
            sensors.put(Sensor.TYPE_LINEAR_ACCELERATION, 1);
            sensors.put(Sensor.TYPE_ROTATION_VECTOR, 2);
        }
        sensorManager = (SensorManager)
                context.getSystemService(Context.SENSOR_SERVICE);
        sensorListener = new SensorEventListener() {
            @Override
            public void onSensorChanged(SensorEvent event) {
                feedSensorData(sensors.get(event.sensor.getType()), event);
            }
            @Override
            public void onAccuracyChanged(Sensor sensor, int accuracy) {}
        };
        // alloca buffer per input e output del modello
        inputBuffer = new float[NUM_TIMESTEPS][NUM_SENSORS * NUM_AXES];
        outputBuffer = new float[1][NUM_CLASSES];
    }

    /**
     * Avvia il rilevamento dell'attività. Da chiamare quando l'app viene
     * avviata la prima volta o riaperta.
     */
    public void start() {
        for (int sensorType: sensors.keySet()) {
            Log.d(TAG, "registering sensor " + sensorType);
            Sensor sensor = sensorManager.getDefaultSensor(sensorType);
            sensorManager.registerListener(sensorListener, sensor,
                    SensorManager.SENSOR_DELAY_GAME);
        }
        resetInputBuffer();
    }

    /**
     * Sospende il rilevamento dell'attività, lasciando allocate le risorse
     * necessarie. Da chiamare se l'app viene messa in pausa (es. passaggio ad
     * altra app o standby).
     */
    public void stop() {
        sensorManager.unregisterListener(sensorListener);
    }

    /**
     * Termina il riconoscitore di attività rilasciando tutte le risorse
     * allocate. Da chiamare all'arresto dell'app.
     */
    public void close() {
        model.close();
        model = null;
        callback = null;
        sensorManager = null;
    }

    /**
     * Metodo chiamato per registrare i dati mandati da un sensore.
     * @param sensor numero del sensore
     * @param event evento generato dal sensore contenente i dati
     */
    void feedSensorData(int sensor, SensorEvent event) {
        // stop se ho gia raccolto dati da questo sensore e ne mancano altri
        if ((inputFilledMask & (1 << sensor)) > 0) {
            return;
        }
        // copia i dati del sensore nel buffer
        System.arraycopy(event.values, 0, inputBuffer[inputFillStep],
                NUM_AXES * sensor, NUM_AXES);
        /*for (int i=0; i<NUM_AXES; i++) {
            // correzione unità da m/s^2 a G (solo accelerometro!)
            inputBuffer[inputFillStep][i] = event.values[i] / SensorManager.GRAVITY_EARTH;
        }*/
        // segna che i dati di questo sensore sono ricevuti
        inputFilledMask |= 1 << sensor;
        // se ho ricevuto dati da tutti i sensori...
        if (inputFilledMask == (1 << NUM_SENSORS) - 1) {
            // avanza il buffer al prossimo passo
            inputFillStep++;
            inputFilledMask = 0;
            // se sono arrivato al termine del buffer...
            if (inputFillStep == NUM_TIMESTEPS) {
                // esegui l'inferenza, quindi azzera il buffer di input
                inferActivity();
                resetInputBuffer();
            } else {
                if (callback != null) {
                    callback.bufferUpdated(inputFillStep, NUM_TIMESTEPS, event.values);
                }
            }
        }
    }

    /**
     * Esegue il riconoscimento dell'attività dai dati presenti nel buffer.
     */
    void inferActivity() {
        // eseguo l'inferenza (si assume che inputBuffer sia pieno)
        model.run(inputBuffer, outputBuffer);
        // in outputBuffer ho ora le probabilità di ciascuna attività
        // verifico quale sia quella con probabilità maggiore
        int output = 0;
        float outputProb = outputBuffer[0][0];
        for (int i=1; i<NUM_CLASSES; i++) {
            if (outputBuffer[0][i] > outputProb) {
                output = i;
                outputProb = outputBuffer[0][i];
            }
        }
        // invio notifica dell'attività corrente
        if (callback != null) {
            callback.activityInferred(output);
        }
    }

    /**
     * Reimposta lo stato del buffer.
     */
    void resetInputBuffer() {
        inputFilledMask = 0;
        inputFillStep = 0;
        if (callback != null) {
            callback.bufferUpdated(0, NUM_TIMESTEPS, null);
        }
    }

    static MappedByteBuffer loadModelFile(Context context) throws IOException {
        AssetFileDescriptor fileDescriptor = context.getAssets().openFd(MODEL_ASSET);
        FileInputStream inputStream =
                new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY,
                startOffset, declaredLength);
    }

}
