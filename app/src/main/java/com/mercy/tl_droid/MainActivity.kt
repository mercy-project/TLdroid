package com.mercy.tl_droid

import android.graphics.drawable.BitmapDrawable
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import com.mercy.classification.Classifier
import kotlinx.android.synthetic.main.activity_main.*

class MainActivity : AppCompatActivity() {

    private var mClassifier: Classifier? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        mClassifier = Classifier.create(this, Classifier.Device.CPU, 1)

        val drawable = resources.getDrawable(R.drawable.lenna, null)
        val bitmap = (drawable as BitmapDrawable).bitmap

        val results = mClassifier?.recognizeImage(bitmap)
        tv_result.text = results.toString()
    }

    override fun onStop() {
        super.onStop()

        mClassifier?.close()
    }
}