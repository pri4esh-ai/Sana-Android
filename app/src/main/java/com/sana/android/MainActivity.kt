package com.sana.android

import android.content.Context
import android.net.Uri
import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.sana.android.engine.NativeSana
import java.io.File
import java.util.concurrent.Executors

class MainActivity : ComponentActivity() {

    override fun onCreate(
        savedInstanceState: Bundle?
    ) {

        super.onCreate(
            savedInstanceState
        )

        setContent {
            SanaTestApp(
                context = this
            )
        }
    }
}


@Composable
private fun SanaTestApp(
    context: Context
) {

    var transformerFile by remember {
        mutableStateOf<File?>(null)
    }

    var vaeFile by remember {
        mutableStateOf<File?>(null)
    }

    var testing by remember {
        mutableStateOf(false)
    }

    var result by remember {
        mutableStateOf(
            "No test performed yet."
        )
    }

    var transformerName by remember {
        mutableStateOf(
            "Transformer not selected"
        )
    }

    var vaeName by remember {
        mutableStateOf(
            "VAE decoder not selected"
        )
    }

    val executor =
        remember {
            Executors.newSingleThreadExecutor()
        }

    DisposableEffect(Unit) {

        onDispose {
            executor.shutdown()
        }
    }

    val transformerPicker =
        rememberLauncherForActivityResult(
            contract =
                ActivityResultContracts.OpenDocument()
        ) { uri: Uri? ->

            if (uri != null) {

                result =
                    "Copying Transformer..."

                executor.execute {

                    try {

                        val file =
                            copyModel(
                                context,
                                uri,
                                "sana_transformer.mnn"
                            )

                        transformerFile =
                            file

                        transformerName =
                            file.name

                        result =
                            "Transformer ready:\n" +
                            file.length() /
                            (1024L * 1024L) +
                            " MB"

                    } catch (e: Exception) {

                        result =
                            "Transformer copy failed:\n" +
                            e.message
                    }
                }
            }
        }


    val vaePicker =
        rememberLauncherForActivityResult(
            contract =
                ActivityResultContracts.OpenDocument()
        ) { uri: Uri? ->

            if (uri != null) {

                result =
                    "Copying VAE..."

                executor.execute {

                    try {

                        val file =
                            copyModel(
                                context,
                                uri,
                                "sana_vae_decoder.mnn"
                            )

                        vaeFile =
                            file

                        vaeName =
                            file.name

                        result =
                            "VAE ready:\n" +
                            file.length() /
                            (1024L * 1024L) +
                            " MB"

                    } catch (e: Exception) {

                        result =
                            "VAE copy failed:\n" +
                            e.message
                    }
                }
            }
        }


    Surface(
        modifier =
            Modifier.fillMaxSize()
    ) {

        Column(
            modifier =
                Modifier
                    .fillMaxSize()
                    .verticalScroll(
                        rememberScrollState()
                    )
                    .padding(20.dp),
            verticalArrangement =
                Arrangement.Top
        ) {

            Text(
                text = "Sana 0.6B Test",
                style =
                    MaterialTheme.typography
                        .headlineMedium
            )

            Spacer(
                modifier =
                    Modifier.height(6.dp)
            )

            Text(
                text =
                    "512 × 512 • MNN • OpenCL / FP16"
            )

            Spacer(
                modifier =
                    Modifier.height(20.dp)
            )

            Card(
                modifier =
                    Modifier.fillMaxWidth()
            ) {

                Column(
                    modifier =
                        Modifier.padding(16.dp)
                ) {

                    Text(
                        text =
                            "1. Transformer"
                    )

                    Spacer(
                        modifier =
                            Modifier.height(6.dp)
                    )

                    Text(
                        text =
                            transformerName
                    )

                    Spacer(
                        modifier =
                            Modifier.height(10.dp)
                    )

                    Button(
                        onClick = {
                            transformerPicker.launch(
                                arrayOf(
                                    "application/octet-stream",
                                    "application/*",
                                    "*/*"
                                )
                            )
                        },
                        enabled = !testing,
                        modifier =
                            Modifier.fillMaxWidth()
                    ) {

                        Text(
                            "Select Transformer"
                        )
                    }
                }
            }

            Spacer(
                modifier =
                    Modifier.height(14.dp)
            )

            Card(
                modifier =
                    Modifier.fillMaxWidth()
            ) {

                Column(
                    modifier =
                        Modifier.padding(16.dp)
                ) {

                    Text(
                        text =
                            "2. VAE Decoder"
                    )

                    Spacer(
                        modifier =
                            Modifier.height(6.dp)
                    )

                    Text(
                        text =
                            vaeName
                    )

                    Spacer(
                        modifier =
                            Modifier.height(10.dp)
                    )

                    Button(
                        onClick = {
                            vaePicker.launch(
                                arrayOf(
                                    "application/octet-stream",
                                    "application/*",
                                    "*/*"
                                )
                            )
                        },
                        enabled = !testing,
                        modifier =
                            Modifier.fillMaxWidth()
                    ) {

                        Text(
                            "Select VAE"
                        )
                    }
                }
            }

            Spacer(
                modifier =
                    Modifier.height(18.dp)
            )

            Row(
                modifier =
                    Modifier.fillMaxWidth()
            ) {

                Button(
                    onClick = {

                        val transformer =
                            transformerFile

                        val vae =
                            vaeFile

                        if (
                            transformer == null ||
                            vae == null
                        ) {

                            Toast.makeText(
                                context,
                                "Select both models first.",
                                Toast.LENGTH_SHORT
                            ).show()

                            return@Button
                        }

                        testing =
                            true

                        result =
                            "Starting native MNN test...\n\n" +
                            "Transformer will run first.\n" +
                            "VAE will run after Transformer is released."

                        executor.execute {

                            val output =
                                try {

                                    NativeSana.testModels(
                                        context,
                                        transformer,
                                        vae,
                                        true
                                    )

                                } catch (
                                    e: Throwable
                                ) {

                                    "NATIVE TEST CRASH/ERROR:\n" +
                                    e.stackTraceToString()
                                }

                            runOnUiThread {

                                result =
                                    output

                                testing =
                                    false
                            }
                        }

                    },
                    enabled =
                        !testing &&
                        transformerFile != null &&
                        vaeFile != null,
                    modifier =
                        Modifier.weight(1f)
                ) {

                    if (testing) {

                        CircularProgressIndicator(
                            modifier =
                                Modifier
                                    .width(22.dp)
                                    .height(22.dp)
                    } else {

                        Text(
                            "TEST SANA"
                        )
                    }
                }

                Spacer(
                    modifier =
                        Modifier.width(8.dp)
                )

                OutlinedButton(
                    onClick = {

                        result =
                            "Ready."

                    },
                    enabled = !testing
                ) {

                    Text(
                        "Clear"
                    )
                }
            }

            Spacer(
                modifier =
                    Modifier.height(18.dp)
            )

            Text(
                text = "Test result",
                style =
                    MaterialTheme.typography
                        .titleMedium
            )

            Spacer(
                modifier =
                    Modifier.height(8.dp)
            )

            Card(
                modifier =
                    Modifier.fillMaxWidth()
            ) {

                Text(
                    text =
                        result,
                    modifier =
                        Modifier
                            .padding(16.dp)
                )
            }

            Spacer(
                modifier =
                    Modifier.height(20.dp)
            )

            Text(
                text =
                    "The first test uses zero-filled tensors. " +
                    "It verifies that the converted MNN graphs " +
                    "can load and execute. It is not yet " +
                    "text-to-image generation."
            )
        }
    }
}


private fun copyModel(
    context: Context,
    uri: Uri,
    fileName: String
): File {

    val modelDirectory =
        File(
            context.filesDir,
            "sana_models"
        )

    if (!modelDirectory.exists()) {
        modelDirectory.mkdirs()
    }

    val destination =
        File(
            modelDirectory,
            fileName
        )

    context.contentResolver
        .openInputStream(uri)
        .use { input ->

            requireNotNull(input) {
                "Unable to open selected file."
            }

            destination.outputStream()
                .use { output ->

                    input.copyTo(
                        output,
                        bufferSize = 1024 * 1024
                    )
                }
        }

    return destination
}
