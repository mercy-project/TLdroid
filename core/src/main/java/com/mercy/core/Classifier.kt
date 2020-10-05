package com.mercy.core

import android.app.Activity
import android.graphics.Bitmap
import android.graphics.RectF
import android.os.SystemClock
import android.os.Trace
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.BufferedReader
import java.io.FileInputStream
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.*
import kotlin.Comparator
import kotlin.collections.ArrayList
import kotlin.math.min

abstract class Classifier {

    companion object {
        private val TAG = Classifier::class.java.simpleName

        /** Number of results to show in the UI. */
        private const val MAX_RESULTS: Int = 3
        /** Dimensions of inputs. */
        private const val DIM_BATCH_SIZE: Int = 1
        private const val DIM_PIXEL_SIZE: Int = 3

        /**
         * Creates a classifier with the provided configuration.
         *
         * @param activity The current Activity.
         * @param model The model to use for classification.
         * @param device The device to use for classification.
         * @param numThreads The number of threads to use for classification.
         * @return A classifier with the desired configuration.
         */
        fun create(activity: Activity, model: Model, device: Device, numThreads: Int): Classifier
                = when (model) {
            Model.QUANTIZED -> ClassifierQuantizedMobileNet(activity, device, numThreads)
            else -> ClassifierFloatMobileNet(activity, device, numThreads)
        }

        /** An immutable result returned by a Classifier describing what was recognized. */
        class Recognition(val id: String,       // A unique identifier for what has been recognized. Specific to the class, not the instance of the object.
                          val title: String,    // Display name for the recognition.
                          val confidence: Float,// A sortable score for how good the recognition is relative to others. Higher should be better.
                          var location: RectF? = null) {// Optional location within the source image for the location of the recognized object.

            override fun toString(): String = "[$id] $title ${String.format("(%.1f%%) ", confidence * 100f)} $location"
        }
    }

    protected constructor(activity: Activity, device: Device, numThreads: Int) {
        val model = loadModelFile(activity).also {
            tfliteModel = it
        }
        when (device) {
            Device.NNAPI -> tfliteOptions.setUseNNAPI(true)
            Device.GPU -> gpuDelegate = GpuDelegate().also {
                tfliteOptions.addDelegate(it)
            }
            else -> { /* Device.CPU */ }
        }
        tfliteOptions.setNumThreads(numThreads)
        tflite = Interpreter(model, tfliteOptions)
        labels = loadLabelList(activity)
        imgData = ByteBuffer.allocateDirect(DIM_BATCH_SIZE * getImageSizeX() * getImageSizeY()
                * DIM_PIXEL_SIZE * getNumBytesPerChannel())
            .apply {
                order(ByteOrder.nativeOrder())
            }
        Log.d(TAG, "Created a TensorFlow Lite Image Classifier.")
    }

    /** The model type used for classification. */
    enum class Model {
        FLOAT, QUANTIZED
    }

    /** The runtime device type used for executing classification. */
    enum class Device {
        CPU, NNAPI, GPU
    }

    /** Preallocated buffers for storing image data in. */
    private val intValues = IntArray(getImageSizeX() * getImageSizeY())

    /** Options for configuring the Interpreter. */
    private val tfliteOptions: Interpreter.Options = Interpreter.Options()

    /** The loaded TensorFlow Lite model. */
    private var tfliteModel: MappedByteBuffer? = null

    /** Labels corresponding to the output of the vision model. */
    private val labels: List<String>

    /** Optional GPU delegate for acceleration. */
    private var gpuDelegate: GpuDelegate? = null

    /** An instance of the driver class to run model inference with TensorFlow Lite. */
    protected var tflite: Interpreter? = null

    /** A ByteBuffer to hold image data, to be feed into TensorFlow Lite as inputs. */
    protected var imgData: ByteBuffer

    /** Reads label list from Assets. */
    private fun loadLabelList(activity: Activity): List<String> {
        val labels = ArrayList<String>()
        val reader: BufferedReader = BufferedReader(InputStreamReader(activity.assets.open(getLabelPath())))
        while (true) {
            val line = reader.readLine() ?: break
            labels.add(line)
        }
        reader.close()
        return labels
    }

    /** Memory-map the model file in Assets. */
    private fun loadModelFile(activity: Activity): MappedByteBuffer {
        val fileDescriptor = activity.assets.openFd(getModelPath())
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset: Long = fileDescriptor.startOffset
        val declaredLength: Long = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    /** Writes Image data into a {@code ByteBuffer}. */
    private fun convertBitmapToByteBuffer(bitmap: Bitmap) {
        imgData?.run {
            this.rewind()
            bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
            // Convert the image to floating point.
            var pixel = 0
            val startTime = SystemClock.uptimeMillis()
            for (i in 0 until getImageSizeX()) {
                for (j in 0 until getImageSizeY()) {
                    val value = intValues[pixel++]
                    addPixelValue(value)
                }
            }
            val endTime = SystemClock.uptimeMillis()
            Log.v(TAG, "Timecost to put values into ByteBuffer: ${endTime - startTime}")
        }
    }

    /** Runs inference and returns the classification results. */
    fun recognizeImage(bitmap: Bitmap): List<Recognition> {
        // Log this method so that it can be analyzed with systrace.
        Trace.beginSection("recognizeImage")

        Trace.beginSection("preprocessBitmap")
        convertBitmapToByteBuffer(bitmap)
        Trace.endSection()

        // Run the inference call.
        Trace.beginSection("runInference")
        val startTime = SystemClock.uptimeMillis()
        runInference()
        val endTime = SystemClock.uptimeMillis()
        Trace.endSection()
        Log.v(TAG, "Timecost to run model inference: ${endTime - startTime}")

        // Find the best classifications.
        val pq = PriorityQueue<Recognition>(3, object: Comparator<Recognition> {
            override fun compare(lhs: Recognition, rhs: Recognition): Int {
                return rhs.confidence.compareTo(lhs.confidence)
            }
        })
        for (i in labels.indices) {
            pq.add(Recognition(i.toString(),
                if (labels.size > i) labels[i] else "unknown",
                getNormalizedProbability(i),
                null))
        }
        val recognitions = ArrayList<Recognition>()
        val recognitionSize = min(pq.size, MAX_RESULTS)
        for (i in 0 until recognitionSize) {
            recognitions.add(pq.poll())
        }
        Trace.endSection()
        return recognitions
    }

    /** Closes the interpreter and model to release resources. */
    fun close() {
        tflite?.apply {
            close()
            tflite = null
        }
        gpuDelegate?.apply {
            close()
            gpuDelegate = null
        }
        tfliteModel = null
    }

    /**
     * Get the image size along the x axis.
     *
     * @return
     */
    abstract fun getImageSizeX(): Int

    /**
     * Get the image size along the y axis.
     *
     * @return
     */
    abstract fun getImageSizeY(): Int

    /**
     * Get the name of the model file stored in Assets.
     *
     * @return
     */
    protected abstract fun getModelPath(): String

    /**
     * Get the name of the label file stored in Assets.
     *
     * @return
     */
    protected abstract fun getLabelPath(): String

    /**
     * Get the number of bytes that is used to store a single color channel value.
     *
     * @return
     */
    protected abstract fun getNumBytesPerChannel(): Int

    /**
     * Add pixelValue to byteBuffer.
     *
     * @param pixelValue
     */
    protected abstract fun addPixelValue(pixelValue: Int)

    /**
     * Read the probability value for the specified label This is either the original value as it was
     * read from the net's output or the updated value after the filter was applied.
     *
     * @param labelIndex
     * @return
     */
    protected abstract fun getProbability(labelIndex: Int): Float

    /**
     * Set the probability value for the specified label.
     *
     * @param labelIndex
     * @param value
     */
    protected abstract fun setProbability(labelIndex: Int, value: Number)

    /**
     * Get the normalized probability value for the specified label. This is the final value as it
     * will be shown to the user.
     *
     * @return
     */
    protected abstract fun getNormalizedProbability(labelIndex: Int): Float

    /**
     * Run inference using the prepared input in {@link #imgData}. Afterwards, the result will be
     * provided by getProbability().
     *
     * <p>This additional method is necessary, because we don't have a common base for different
     * primitive data types.
     */
    protected abstract fun runInference()

    /**
     * Get the total number of labels.
     *
     * @return
     */
    protected fun getNumLabels(): Int = labels.size
}