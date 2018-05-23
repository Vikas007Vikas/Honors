/*
 *    Copyright 2016 Anand Muralidhar
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package com.anandmuralidhar.simplearandroid;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.DialogInterface;
import android.content.res.AssetManager;
import android.hardware.Camera;
import android.opengl.GLSurfaceView;
import android.os.Bundle;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.widget.FrameLayout;



public class SimpleARActivity extends Activity{

    private GLSurfaceView mGLView = null;
    //temp
    //private Camera mFrontCamera;
    //private MyFrontSurfaceView mFrontCamPreview;

    private CameraClass mCameraObject_back;
    //private CameraClass mFrontCamera;
    private boolean appIsExiting=false;
    private GestureClass mGestureObject;
    private SensorClass mSensorObject;

    private static int cameraIndex = Camera.CameraInfo.CAMERA_FACING_BACK;

    private native void CreateObjectNative(AssetManager assetManager, String pathToInternalDir);
    private native void DeleteObjectNative();
    private native void SetCameraParamsNative(int previewWidth, int previewHeight, float cameraFOV);

    public static String TAG = "SimpleARActivity";
    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        AssetManager assetManager = getAssets();
        String pathToInternalDir = getFilesDir().getAbsolutePath();

        mCameraObject_back = new CameraClass(cameraIndex);
        if(!mCameraObject_back.IsResolutionSupported()) {
            ShowExitDialog(this, getString(R.string.exit_no_resolution));
            return;
        }

        // call the native constructors to create an object
        CreateObjectNative(assetManager, pathToInternalDir);
        SetCameraParamsNative(mCameraObject_back.GetPreviewWidth(), mCameraObject_back.GetPreviewHeight(),
                mCameraObject_back.GetFOV());

        // layout has only two components, a GLSurfaceView and a TextView
        setContentView(R.layout.simplear_layout);
        mGLView = (MyGLSurfaceView) findViewById (R.id.gl_surface_view);

        // mGestureObject will handle touch gestures on the screen
        mGestureObject = new GestureClass(this);
        mGLView.setOnTouchListener(new View.OnTouchListener() {
            @Override
            public boolean onTouch(View v, MotionEvent event) {
                mGestureObject.mTapDetector.onTouchEvent(event);
                return true;
            }
        });

        mSensorObject = new SensorClass(this, mGLView);
        if(!mSensorObject.isSensorsAvailable()) {
            ShowExitDialog(this, getResources().getString(R.string.exit_no_sensor));
            appIsExiting=true;
        }

        //adding temporarily -----------------------------------------------------------------
//        mFrontCamera = new CameraClass(cameraIndex);
//        if(!mFrontCamera.IsResolutionSupported()) {
//            ShowExitDialog(this, getString(R.string.exit_no_resolution));
//            return;
//        }
//
//        // call the native constructors to create an object
//        CreateObjectNative(assetManager, pathToInternalDir);
//        SetCameraParamsNative(mFrontCamera.GetPreviewWidth(), mFrontCamera.GetPreviewHeight(),
//                mFrontCamera.GetFOV());


//        mFrontCamera = getCameraInstance(1);
//        mFrontCamPreview = new MyFrontSurfaceView(this, mFrontCamera);
//        FrameLayout frontPreview = (FrameLayout) findViewById(R.id.front_camera_preview);
//        frontPreview.addView(mFrontCamPreview);

        //-------------------------------------------------------------------------------------



    }

    @Override
    protected void onResume() {

        super.onResume();

        if(appIsExiting) {
            return;
        }

        // Android suggests that we call onResume on GLSurfaceView
        if (mGLView != null) {
            mGLView.onResume();
        }


        if(!mSensorObject.RegisterSensors()){
            ShowExitDialog(this, getResources().getString(R.string.exit_no_reg_sensor));
            appIsExiting=true;
            return;
        }

        // initialize the camera again in case activity was paused and resumed
        mCameraObject_back.InitializeCamera();
        mCameraObject_back.StartCamera();

    }

    @Override
    protected void onPause() {

        super.onPause();

        // Android suggests that we call onPause on GLSurfaceView
        if(mGLView != null) {
            mGLView.onPause();
        }

        mSensorObject.UnregisterSensors();

        mCameraObject_back.StopCamera();

    }

    @Override
    protected void onDestroy() {

        super.onDestroy();

        // We are exiting the activity, let's delete the native objects
        DeleteObjectNative();

    }

    public void ShowExitDialog(final Activity activity, String exitMessage){
        appIsExiting = true;
        AlertDialog.Builder alertDialogBuilder = new AlertDialog.Builder(activity);
        alertDialogBuilder.setMessage(exitMessage)
                .setCancelable(false)
                .setPositiveButton("OK", new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface arg0, int arg1) {
                        activity.finish();
                    }
                });

        AlertDialog alertDialog = alertDialogBuilder.create();
        alertDialog.show();
    }

    public static Camera getCameraInstance(int cameraId){
        Camera c = null;
        try {
            c = Camera.open(cameraId); // attempt to get a Camera instance
        }
        catch (Exception e){
            // Camera is not available (in use or does not exist)
            Log.e(TAG,"Camera " + cameraId + " not available! " + e.toString() );
        }
        return c; // returns null if camera is unavailable
    }

    /**
     * load libSimpleARNative.so since it has all the native functions
     */
    static {
        System.loadLibrary("SimpleARNative");
    }
}
