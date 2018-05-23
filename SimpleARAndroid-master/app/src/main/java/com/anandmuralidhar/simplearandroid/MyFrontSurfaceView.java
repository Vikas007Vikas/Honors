package com.anandmuralidhar.simplearandroid;

import java.io.IOException;

import android.app.Activity;
import android.content.Context;
import android.graphics.Matrix;
import android.graphics.RectF;
import android.hardware.Camera;
import android.util.Log;
import android.view.SurfaceHolder;
import android.view.SurfaceView;

/** A basic Camera preview class */
public class MyFrontSurfaceView extends SurfaceView implements SurfaceHolder.Callback {
    private SurfaceHolder mHolder;
    private Camera mCamera;
    private FaceOverlayView mFaceView;
    private Camera.Face[] faces;
    private Camera.Face[] mFaces;

    public static String TAG = "FrontCameraPreview";

    /**
     * Sets the faces for the overlay view, so it can be updated
     * and the face overlays will be drawn again.
     */
    private Camera.FaceDetectionListener faceDetectionListener = new Camera.FaceDetectionListener() {
        @Override
        public void onFaceDetection(Camera.Face[] faces, Camera camera) {
            Log.d("onFaceDetection", "Number of Faces:" + faces.length);
            // Update the view now!
            //mFaceView.setFaces(faces);
            mFaces = faces;
            Log.d("onFaceDetection", "BNumber of Faces:" + mFaces.length);
            if (mFaces != null && mFaces.length > 0) {
                Log.d("abc","length "+mFaces.length);
                Matrix matrix = new Matrix();
                matrix.setScale(true ? -1 : 1, 1);
                matrix.postScale(getWidth() / 2000f, getHeight() / 2000f);
                matrix.postTranslate(getWidth() / 2f, getHeight() / 2f);
                RectF rectF = new RectF();
                for (Camera.Face face : mFaces) {
                    rectF.set(face.rect);
                    matrix.mapRect(rectF);
                    Log.d("abc","relative coordinates "+ rectF.right + " " + rectF.top);
                    // 3d_face_coordinates = calculate3dpoints(rectF.right,rectF.top);
                }
            }
        }
    };

    public MyFrontSurfaceView(Context context, Camera camera) {
        super(context);
        mCamera = camera;

        // Now create the OverlayView:
        mFaceView = new FaceOverlayView(context);

        // Install a SurfaceHolder.Callback so we get notified when the
        // underlying surface is created and destroyed.
        mHolder = getHolder();
        mHolder.addCallback(this);
        // deprecated setting, but required on Android versions prior to 3.0
        mHolder.setType(SurfaceHolder.SURFACE_TYPE_PUSH_BUFFERS);
    }

    public void surfaceCreated(SurfaceHolder holder) {
        // The Surface has been created, now tell the camera where to draw the preview.
        mCamera.setFaceDetectionListener(faceDetectionListener);
        mCamera.startFaceDetection();
        Log.d("onFaceDetection", "Number of Faces:");
        try {
            mCamera.setPreviewDisplay(holder);
            mCamera.startPreview();
        } catch (IOException e) {
            Log.d(TAG, "Error setting camera preview: " + e.getMessage());
        }
    }

    public void surfaceDestroyed(SurfaceHolder holder) {
        // empty. Take care of releasing the Camera preview in your activity.
    }

    public void surfaceChanged(SurfaceHolder holder, int format, int w, int h) {
        // If your preview can change or rotate, take care of those events here.
        // Make sure to stop the preview before resizing or reformatting it.
        Log.d("onFaceDetection", "Number of Faces:");
        if (mHolder.getSurface() == null){
            // preview surface does not exist
            return;
        }

        // stop preview before making changes
        try {
            mCamera.stopPreview();
        } catch (Exception e){
            // ignore: tried to stop a non-existent preview
        }


        // set preview size and make any resize, rotate or
        // reformatting changes here

        // start preview with new settings
        try {
            mCamera.setPreviewDisplay(mHolder);
            mCamera.startPreview();

        } catch (Exception e){
            Log.d(TAG, "Error starting camera preview: " + e.getMessage());
        }
    }
}
