package at.tugraz.icg.dualcamera;

import android.content.Intent;
import android.hardware.Camera;
import android.os.Bundle;
import android.app.Activity;
import android.util.Log;
import android.view.Menu;
import android.widget.FrameLayout;

import com.Sitepoint.Myapp.UnityPlayerActivity;

public class DualCamActivity extends Activity {

	private Camera mBackCamera;
	private Camera mFrontCamera;
    private BackCameraPreview mBackCamPreview;
    private FrontCameraPreview mFrontCamPreview;
    
    public static String TAG = "DualCamActivity";
    
	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		setContentView(R.layout.activity_dual_cam);

        //Intent intent = getPackageManager().getLaunchIntentForPackage("com.Sitepoint.AugmentedApp");
        //startActivity(intent);

		/*Intent intent = new Intent(this, UnityPlayerActivity.class);
		startActivity(intent);*/



		Log.i(TAG, "Number of cameras: " + Camera.getNumberOfCameras());
				
		// Create an instance of Camera
		mBackCamera = getCameraInstance(0);


        // Create back camera Preview view and set it as the content of our activity.
        mBackCamPreview = new BackCameraPreview(this, mBackCamera);
        FrameLayout backPreview = (FrameLayout) findViewById(R.id.back_camera_preview);
        backPreview.addView(mBackCamPreview);
        
		mFrontCamera = getCameraInstance(1);
        mFrontCamPreview = new FrontCameraPreview(this, mFrontCamera);
        FrameLayout frontPreview = (FrameLayout) findViewById(R.id.front_camera_preview);
        frontPreview.addView(mFrontCamPreview);

	}

	@Override
	public boolean onCreateOptionsMenu(Menu menu) {
		// Inflate the menu; this adds items to the action bar if it is present.
		getMenuInflater().inflate(R.menu.dual_cam, menu);
		return true;
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
}