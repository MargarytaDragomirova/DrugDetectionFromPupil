package com.example.myapplication

import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import kotlin.math.PI

class MainActivity : AppCompatActivity() {

    companion object {
        init {
            if (!OpenCVLoader.initDebug()) {
                throw RuntimeException("OpenCV failed to initialize")
            }
        }

        const val REQUEST_IMAGE_CAPTURE = 1
    }

    private lateinit var imageView: ImageView
    private lateinit var captureButton: Button
    private lateinit var resultText: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        imageView = findViewById(R.id.imageView)
        captureButton = findViewById(R.id.captureButton)
        resultText = findViewById(R.id.resultText)

        captureButton.setOnClickListener {
            val intent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
            startActivityForResult(intent, REQUEST_IMAGE_CAPTURE)
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        if (resultCode == Activity.RESULT_OK) {
            val bitmap: Bitmap? = when (requestCode) {
                REQUEST_IMAGE_CAPTURE -> data?.extras?.get("data") as? Bitmap
                else -> null
            }

            bitmap?.let {
                val processedBitmap = processImage(it)
                imageView.setImageBitmap(processedBitmap)
            }
        }
    }

    private fun processImage(bitmap: Bitmap): Bitmap {
        val srcMat = Mat()
        Utils.bitmapToMat(bitmap, srcMat)

        val gray = Mat()
        Imgproc.cvtColor(srcMat, gray, Imgproc.COLOR_RGBA2GRAY)
        Imgproc.GaussianBlur(gray, gray, Size(7.0, 7.0), 0.0)

        val minLoc = Core.minMaxLoc(gray).minLoc
        val seedPoint = Point(minLoc.x, minLoc.y)

        val mask = Mat.zeros(gray.rows() + 2, gray.cols() + 2, CvType.CV_8UC1)
        Imgproc.floodFill(gray.clone(), mask, seedPoint, Scalar(255.0), null, Scalar(20.0), Scalar(20.0), Imgproc.FLOODFILL_FIXED_RANGE)

        val pupilMask = Mat()
        Core.inRange(mask.submat(Rect(1, 1, gray.cols(), gray.rows())), Scalar(1.0), Scalar(255.0), pupilMask)

        val pupilContours = mutableListOf<MatOfPoint>()
        Imgproc.findContours(pupilMask, pupilContours, Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)

        var pupilRadius = 0.0
        var pupilCenter = Point()
        pupilContours.maxByOrNull { Imgproc.contourArea(it) }?.let {
            val points = MatOfPoint2f(*it.toArray())
            val radius = FloatArray(1)
            Imgproc.minEnclosingCircle(points, pupilCenter, radius)
            pupilRadius = radius[0].toDouble()
            Imgproc.circle(srcMat, pupilCenter, pupilRadius.toInt(), Scalar(0.0, 255.0, 0.0), 2)
        }

        val irisResult = detectIris(gray, pupilCenter, pupilRadius)
        var irisRadius = 0.0
        var irisCenter = pupilCenter
        if (irisResult != null) {
            irisCenter = irisResult.first
            irisRadius = irisResult.second.toDouble()
            Imgproc.circle(srcMat, irisCenter, irisRadius.toInt(), Scalar(255.0, 0.0, 0.0), 2)
        }

        val pupilArea = PI * (pupilRadius * pupilRadius)
        val irisArea = PI * (irisRadius * irisRadius)
        val ratio = if (irisArea > 0) (pupilArea / irisArea * 100).coerceAtMost(100.0) else 0.0
        val status = when {
            ratio > 30 -> "Pupil too large (possible drug effect)"
            ratio < 5 -> "Pupil too small (possible drug effect)"
            else -> "Pupil size normal"
        }

        runOnUiThread {
            resultText.text = "$status".format(ratio)
        }

        val resultBitmap = Bitmap.createBitmap(srcMat.cols(), srcMat.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(srcMat, resultBitmap)
        return resultBitmap
    }

    private fun detectIris(srcGray: Mat, pupilCenter: Point, pupilRadius: Double): Pair<Point, Float>? {
        if (pupilRadius <= 0) return null

        val maxRadius = (Math.min(srcGray.cols(), srcGray.rows()) / 2).toInt()
        val step = 1
        val gradientMagnitudes = mutableListOf<Pair<Int, Double>>()

        val blurred = Mat()
        Imgproc.medianBlur(srcGray, blurred, 5)

        val gradX = Mat()
        val gradY = Mat()
        Imgproc.Sobel(blurred, gradX, CvType.CV_64F, 1, 0, 3)
        Imgproc.Sobel(blurred, gradY, CvType.CV_64F, 0, 1, 3)

        for (r in (pupilRadius.toInt() + 5) until maxRadius step step) {
            var sumGrad = 0.0
            val numPoints = 360
            for (angleDeg in 0 until numPoints) {
                val angle = Math.toRadians(angleDeg.toDouble())
                val x = (pupilCenter.x + r * Math.cos(angle)).toInt()
                val y = (pupilCenter.y + r * Math.sin(angle)).toInt()
                if (x < 0 || y < 0 || x >= srcGray.cols() || y >= srcGray.rows()) continue

                val gx = gradX.get(y, x)[0]
                val gy = gradY.get(y, x)[0]
                val gradMag = Math.sqrt(gx * gx + gy * gy)
                sumGrad += gradMag
            }
            val avgGrad = sumGrad / numPoints
            gradientMagnitudes.add(Pair(r, avgGrad))
        }

        val irisRadiusCandidate = gradientMagnitudes.maxByOrNull { it.second }?.first ?: return null


        return Pair(pupilCenter, irisRadiusCandidate.toFloat())
    }
}