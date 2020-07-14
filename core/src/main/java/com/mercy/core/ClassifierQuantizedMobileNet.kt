package com.mercy.core

import android.app.Activity

class ClassifierQuantizedMobileNet(activity: Activity, device: Device, numThreads: Int)
    : Classifier(activity, device, numThreads) {

    /**
     * An array to hold inference results, to be feed into Tensorflow Lite as outputs. This isn't part
     * of the super class, because we need a primitive array here.
     */
    private val labelProbArray: Array<ByteArray> = arrayOf(ByteArray(getNumLabels()))

    override fun getImageSizeX(): Int = 224

    override fun getImageSizeY(): Int = 224

    override fun getModelPath(): String = "mobilenet_v1_1.0_224_quant.tflite"

    override fun getLabelPath(): String = "mobilenet_v1_1.0_224_labels.txt"

    // the quantized model uses a single byte only
    override fun getNumBytesPerChannel(): Int = 1

    override fun addPixelValue(pixelValue: Int) {
        imgData.put(pixelValue.shr(16).and(0xFF).toByte())
        imgData.put(pixelValue.shr(8).and(0xFF).toByte())
        imgData.put(pixelValue.and(0xFF).toByte())
    }

    override fun getProbability(labelIndex: Int): Float = labelProbArray[0][labelIndex].toFloat()

    override fun setProbability(labelIndex: Int, value: Number) {
        labelProbArray[0][labelIndex] = value.toByte()
    }

    override fun getNormalizedProbability(labelIndex: Int): Float = (labelProbArray[0][labelIndex].toInt().and(0xFF)) / 255f

    override fun runInference() {
        tflite?.run(imgData, labelProbArray)
    }
}