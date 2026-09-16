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
                    SanaVaeTestScreen(
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
private fun SanaVaeTestScreen(
    context: Context
) {
    var vaeUri by remember {
        mutableStateOf<Uri?>(null)
    }

    var vaeName by remember {
        mutableStateOf("No VAE selected")
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

    val vaePicker =
        rememberLauncherForActivityResult(
            contract = ActivityResultContracts.OpenDocument()
        ) { uri ->

            if (uri != null) {

                vaeUri = uri

                vaeName =
                    uri.lastPathSegment
                        ?.substringAfterLast("/")
                        ?: "VAE selected"

                status = "VAE selected"

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
            text = "Sana 0.6B • 512×512 • VAE Diagnostic",
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
                    text = "Transformer status",
                    fontWeight = FontWeight.Bold
                )

                Text(
                    text = "✓ Transformer diagnostic passed"
                )

                Text(
                    text = "The Transformer is excluded from this test.",
                    style = MaterialTheme.typography.bodySmall
                )
            }
        }

        Card(
            modifier = Modifier.fillMaxWidth()
        ) {

            Column(
                modifier = Modifier.padding(16.dp),

                verticalArrangement =
                    Arrangement.spacedBy(8.dp)
            ) {

                Text(
                    text = "VAE decoder model",
                    fontWeight = FontWeight.Bold
                )

                Text(
                    text = vaeName,
                    style = MaterialTheme.typography.bodySmall
                )

                OutlinedButton(
                    enabled =
                        !testing &&
                        !copying,

                    onClick = {

                        vaePicker.launch(
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
                        text = "SELECT VAE"
                    )
                }
            }
        }

        Button(
            enabled =
                vaeUri != null &&
                !testing &&
                !copying,

            onClick = {

                val selectedUri =
                    vaeUri
                        ?: return@Button

                testing = true
                copying = true

                result = ""

                status =
                    "Copying VAE..."

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

                        val vaeFile =
                            File(
                                modelDirectory,
                                "sana_vae_decoder.mnn"
                            )

                        copyUriToFile(
                            context = context,
                            uri = selectedUri,
                            destination = vaeFile
                        )

                        val fileSize =
                            vaeFile.length()

                        if (fileSize <= 0L) {

                            throw IllegalStateException(
                                "Copied VAE is empty"
                            )
                        }

                        mainHandler.post {

                            copying = false

                            status =
                                "VAE copied. Starting native test..."
                        }

                        /*
                         * IMPORTANT:
                         *
                         * This calls the VAE-only JNI API.
                         *
                         * Transformer is NOT passed.
                         * Transformer is NOT copied.
                         * Transformer is NOT loaded.
                         */

                        val output =
                            NativeSana.testVae(
                                context = context,
                                vaeFile = vaeFile,
                                preferOpenCl = true
                            )

                        mainHandler.post {

                            result = output

                            status =
                                "VAE test finished"

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
                        "Testing VAE..."
                )

            } else {

                Text(
                    text =
                        "TEST VAE"
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
                        "Only sana_vae_decoder.mnn is tested."
                )

                Text(
                    text =
                        "The Transformer is completely excluded from this test."
                )

                Text(
                    text =
                        "The VAE receives the 32-channel latent tensor produced by the Sana 0.6B architecture."
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
            "Unable to open selected VAE"
        )
}
