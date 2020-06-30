/*
 * Copyright 2020 The Mercy-Project Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.mercy.classification

import android.app.Activity

class MercyVisionClassifier(activity: Activity, device: Device, numThreads: Int)
    : Classifier(activity, device, numThreads) {

    companion object {
        private val TAG = MercyVisionClassifier::class.java.simpleName

        private const val IMAGE_MAX = 255f

    }
    override fun getDimPixelSize(): Int = 4

    private val labelProbArray: Array<FloatArray> = arrayOf(FloatArray(getNumLabels()))

    override fun getImageSizeX(): Int = 48

    override fun getImageSizeY(): Int = 48

    override fun getModelPath(): String = "mercy_vision_v1_1.0_48.tflite"

    override fun getLabelPath(): String = "mercy_vision_v1_1.0_48_info.txt"

    override fun getNumBytesPerChannel(): Int = 1   // Gray scaled

    override fun addPixelValue(pixelValue: Int) {
        // pixelValue: 0bRRRRRRRR_GGGGGGGG_BBBBBBBB
        val grayPixel: Float = (pixelValue.shr(16).and(0xFF)
                + pixelValue.shr(8).and(0xFF)
                + pixelValue.and(0xFF)) / 3f
        imgData.putFloat(grayPixel / IMAGE_MAX)
    }

    override fun getProbability(labelIndex: Int): Float = labelProbArray[0][labelIndex]

    override fun setProbability(labelIndex: Int, value: Number) {
        labelProbArray[0][labelIndex] = value.toFloat()
    }

    override fun getNormalizedProbability(labelIndex: Int): Float = labelProbArray[0][labelIndex]

    override fun runInference() {
        tfLite?.run(imgData, labelProbArray)
    }
}