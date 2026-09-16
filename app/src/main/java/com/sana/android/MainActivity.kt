package com.sana.android

import android.content.Context
import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.sana.android.engine.NativeSana
import java.io.File
import java.util.concurrent.Executors

class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContent {
            MaterialTheme {
                Surface(
                    modifier = Modifier.fillMaxSize()
                ) {
                    SanaTransformerTestScreen(
                        context = this@MainActivity
                    )
                }
            }
        }
    }

    override fun onDestroy() {
        try {
            NativeSana.release()
        } catch (_: Throwable) {
        }

        super.onDestroy()
    }
}

@Composable
private fun SanaTransformerTestScreen(
    context: Context
) {
    var transformerUri by remember {
        mutableStateOf<Uri?>(null)
    }

    var transformerName by remember {
        mutableStateOf("No Transformer selected")
    }

    var status by remember {
        mutableStateOf("Ready")
    }

    var result by remember {
        mutableStateOf("")
    }

    var testing by remember {
        mutableStateOf(false)
    }

    var copying by remember {
        mutableStateOf(false)
    }

    val executor = remember {
        Executors.newSingleThreadExecutor()
    }

    val mainHandler = remember {
        Handler(Looper.getMainLooper())
    }

    DisposableEffect(Unit) {
        onDispose {
            executor.shutdownNow()
        }
    }

    val transformerPicker =
        rememberLauncherForActivityResult(
            contract = ActivityResultContracts.OpenDocument()
        ) { uri ->

            if (uri != null) {

                transformerUri = uri

                transformerName =
                    uri.lastPathSegment
                        ?.substringAfterLast("/")
                        ?: "Transformer selected"

                status = "Transformer selected"

                result = ""
            }
        }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(
                rememberScrollState()
            )
            .padding(20.dp),

        verticalArrangement =
            Arrangement.spacedBy(12.dp)
    ) {

        Text(
            text = "Sana Android",
            style = MaterialTheme.typography.headlineMedium,
            fontWeight = FontWeight.Bold
        )

        Text(
            text = "Sana 0.6B • 512×512 • Transformer Diagnostic",
            style = MaterialTheme.typography.bodyMedium
        )

        Text(
            text = "MNN • ARM64 • OpenCL / FP16",
            style = MaterialTheme.typography.bodySmall
        )

        Spacer(
            modifier = Modifier.height(8.dp)
        )

        Card(
            modifier = Modifier.fillMaxWidth()
        ) {

            Column(
                modifier = Modifier.padding(16.dp),

                verticalArrangement =
                    Arrangement.spacedBy(8.dp)
            ) {

                Text(
                    text = "Transformer model",
                    fontWeight = FontWeight.Bold
                )

                Text(
                    text = transformerName,
                    style = MaterialTheme.typography.bodySmall
                )

                OutlinedButton(
                    enabled =
                        !testing &&
                        !copying,

                    onClick = {

                        transformerPicker.launch(
                            arrayOf(
                                "application/octet-stream",
                                "application/*",
                                "*/*"
                            )
                        )
                    },

                    modifier = Modifier.fillMaxWidth()
                ) {

                    Text(
                        text = "SELECT TRANSFORMER"
                    )
                }
            }
        }

        Button(
            enabled =
                transformerUri != null &&
                !testing &&
                !copying,

            onClick = {

                val selectedUri =
                    transformerUri
                        ?: return@Button

                testing = true
                copying = true

                result = ""

                status =
                    "Copying Transformer..."

                executor.execute {

                    try {

                        val modelDirectory =
                            File(
                                context.filesDir,
                                "sana_models"
                            )

                        if (
                            !modelDirectory.exists() &&
                            !modelDirectory.mkdirs()
                        ) {

                            throw IllegalStateException(
                                "Unable to create model directory"
                            )
                        }

                        val transformerFile =
                            File(
                                modelDirectory,
                                "sana_transformer.mnn"
                            )

                        copyUriToFile(
                            context = context,
                            uri = selectedUri,
                            destination = transformerFile
                        )

                        val fileSize =
                            transformerFile.length()

                        if (fileSize <= 0L) {

                            throw IllegalStateException(
                                "Copied Transformer is empty"
                            )
                        }

                        mainHandler.post {

                            copying = false

                            status =
                                "Transformer copied. Starting native test..."
                        }

                        /*
                         * IMPORTANT:
                         *
                         * This calls the NEW transformer-only JNI API.
                         *
                         * VAE is NOT passed.
                         * VAE is NOT copied.
                         * VAE is NOT loaded.
                         */

                        val output =
                            NativeSana.testTransformer(
                                context = context,
                                transformerFile =
                                    transformerFile,
                                preferOpenCl = true
                            )

                        mainHandler.post {

                            result = output

                            status =
                                "Transformer test finished"

                            testing = false
                        }

                    } catch (t: Throwable) {

                        val message =
                            buildString {

                                append(
                                    t::class.java.simpleName
                                )

                                append(": ")

                                append(
                                    t.message
                                        ?: "Unknown error"
                                )
                            }

                        mainHandler.post {

                            copying = false

                            testing = false

                            status =
                                "Test failed"

                            result =
                                message
                        }
                    }
                }
            },

            modifier =
                Modifier.fillMaxWidth()
        ) {

            if (testing) {

                CircularProgressIndicator(
                    modifier = Modifier
                        .height(22.dp)
                )

                Spacer(
                    modifier =
                        Modifier.height(4.dp)
                )

                Text(
                    text =
                        "Testing Transformer..."
                )

            } else {

                Text(
                    text =
                        "TEST TRANSFORMER"
                )
            }
        }

        Card(
            modifier =
                Modifier.fillMaxWidth()
        ) {

            Column(
                modifier =
                    Modifier.padding(16.dp),

                verticalArrangement =
                    Arrangement.spacedBy(8.dp)
            ) {

                Text(
                    text = "Status",
                    fontWeight = FontWeight.Bold
                )

                Text(
                    text = status
                )

                if (result.isNotBlank()) {

                    Spacer(
                        modifier =
                            Modifier.height(4.dp)
                    )

                    Text(
                        text = "Result",
                        fontWeight =
                            FontWeight.Bold
                    )

                    Text(
                        text = result
                    )
                }
            }
        }

        Card(
            modifier =
                Modifier.fillMaxWidth()
        ) {

            Column(
                modifier =
                    Modifier.padding(16.dp),

                verticalArrangement =
                    Arrangement.spacedBy(6.dp)
            ) {

                Text(
                    text =
                        "Diagnostic mode",
                    fontWeight =
                        FontWeight.Bold
                )

                Text(
                    text =
                        "Only sana_transformer.mnn is tested."
                )

                Text(
                    text =
                        "The VAE is completely excluded from this test."
                )

                Text(
                    text =
                        "After the Transformer passes, we will diagnose the VAE separately."
                )
            }
        }
    }
}

private fun copyUriToFile(
    context: Context,
    uri: Uri,
    destination: File
) {

    context.contentResolver
        .openInputStream(uri)
        ?.use { input ->

            destination.outputStream()
                .use { output ->

                    val buffer =
                        ByteArray(1024 * 1024)

                    while (true) {

                        val read =
                            input.read(buffer)

                        if (read <= 0) {
                            break
                        }

                        output.write(
                            buffer,
                            0,
                            read
                        )
                    }

                    output.flush()
                }

        }
        ?: throw IllegalStateException(
            "Unable to open selected Transformer"
        )
}
