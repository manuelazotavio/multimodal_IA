package com.avtracker.mobile.voice

import android.content.Context
import java.io.File

object AssetFiles {
    /**
     * Copies [assetPath] into the app's files dir and returns the file, reusing the copy from a previous run.
     *
     * A sidecar marker holding the package's last update time tells whether the copy belongs to the installed
     * build. (AssetManager.openFd would give the asset size, but it throws for compressed assets, which is
     * what the APK does to .onnx files.)
     */
    fun materialize(context: Context, assetPath: String): File {
        val target = File(context.filesDir, "assets/$assetPath")
        val marker = File(target.path + ".installed")
        val installedAt = context.packageManager.getPackageInfo(context.packageName, 0).lastUpdateTime.toString()

        if (target.exists() && marker.exists() && marker.readText() == installedAt) return target

        target.parentFile?.mkdirs()
        marker.delete()
        context.assets.open(assetPath).use { input ->
            target.outputStream().use { output -> input.copyTo(output, bufferSize = 1 shl 16) }
        }
        marker.writeText(installedAt)
        return target
    }
}
