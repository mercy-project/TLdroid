package com.mercy.core

import android.app.Activity

class ClassifierFloatMobileNet(activity: Activity, device: Classifier.Device, numThreads: Int)
    : Classifier(activity, device, numThreads) {

    companion object {
        private const val IMAGE_MEAN = 127.5f
        private const val IMAGE_STD = 127.5f
    }

    /**
     * An array to hold inference results, to be feed into Tensorflow Lite as outputs. This isn't part
     * of the super class, because we need a primitive array here.
     */
    private val labelProbArray: Array<FloatArray> = arrayOf(FloatArray(getNumLabels()))

    override fun getImageSizeX(): Int = 224

    override fun getImageSizeY(): Int = 224

    override fun getModelPath(): String = "mobilenet_v1_1.0_224.tflite"

    override fun getLabelPath(): String = "mobilenet_v1_1.0_224_labels.txt"

    override fun getNumBytesPerChannel(): Int = 4   // Float.SIZE / Byte.SIZE

    override fun addPixelValue(pixelValue: Int) {
        imgData.putFloat((pixelValue.shr(16).and(0xFF) - IMAGE_MEAN) / IMAGE_STD)
        imgData.putFloat((pixelValue.shr(8).and(0xFF) - IMAGE_MEAN) / IMAGE_STD)
        imgData.putFloat((pixelValue.and(0xFF) - IMAGE_MEAN) / IMAGE_STD)
    }

    override fun getProbability(labelIndex: Int): Float = labelProbArray[0][labelIndex]

    override fun setProbability(labelIndex: Int, value: Number) {
        labelProbArray[0][labelIndex] = value.toFloat()
    }

    override fun getNormalizedProbability(labelIndex: Int): Float = labelProbArray[0][labelIndex]

    override fun runInference() {
        tflite?.run(imgData, labelProbArray)
    }
}