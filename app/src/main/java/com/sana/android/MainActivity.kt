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
                Surface(modifier = Modifier.fillMaxSize()) {
                    SanaTestScreen(context = this@MainActivity)
                }
            }
        }
    }

    override fun onDestroy() {
        NativeSana.release()
        super.onDestroy()
    }
}

@Composable
private fun SanaTestScreen(context: Context) {

    var transformerUri by remember { mutableStateOf<Uri?>(null) }
    var vaeUri by remember { mutableStateOf<Uri?>(null) }

    var transformerName by remember { mutableStateOf("No Transformer selected") }
    var vaeName by remember { mutableStateOf("No VAE selected") }

    var status by remember { mutableStateOf("Ready") }
    var result by remember { mutableStateOf("") }

    var testing by remember { mutableStateOf(false) }
    var copying by remember { mutableStateOf(false) }

    val executor = remember { Executors.newSingleThreadExecutor() }
    val mainHandler = remember { Handler(Looper.getMainLooper()) }

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
                    uri.lastPathSegment?.substringAfterLast("/")
                        ?: "Transformer selected"
                status = "Transformer selected"
            }
        }

    val vaePicker =
        rememberLauncherForActivityResult(
            contract = ActivityResultContracts.OpenDocument()
        ) { uri ->
            if (uri != null) {
                vaeUri = uri
                vaeName =
                    uri.lastPathSegment?.substringAfterLast("/")
                        ?: "VAE selected"
                status = "VAE selected"
            }
        }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {

        Text(
            text = "Sana Android",
            style = MaterialTheme.typography.headlineMedium,
            fontWeight = FontWeight.Bold
        )

        Text(
            text = "Sana 0.6B • 512×512 • MNN • ARM64",
            style = MaterialTheme.typography.bodyMedium
        )

        Spacer(modifier = Modifier.height(8.dp))

        Card(modifier = Modifier.fillMaxWidth()) {
            Column(
                modifier = Modifier.padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {

                Text(
                    text = "1. Transformer model",
                    fontWeight = FontWeight.Bold
                )

                Text(
                    text = transformerName,
                    style = MaterialTheme.typography.bodySmall
                )

                OutlinedButton(
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
                    Text("Select sana_transformer.mnn")
                }
            }
        }

        Card(modifier = Modifier.fillMaxWidth()) {
            Column(
                modifier = Modifier.padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {

                Text(
                    text = "2. VAE decoder",
                    fontWeight = FontWeight.Bold
                )

                Text(
                    text = vaeName,
                    style = MaterialTheme.typography.bodySmall
                )

                OutlinedButton(
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
                    Text("Select sana_vae_decoder.mnn")
                }
            }
        }

        Spacer(modifier = Modifier.height(4.dp))

        Button(
            enabled = transformerUri != null &&
                    vaeUri != null &&
                    !testing &&
                    !copying,
            onClick = {

                val transformer = transformerUri
                val vae = vaeUri

                if (transformer == null || vae == null) {
                    status = "Select both models first"
                    return@Button
                }

                testing = true
                copying = true
                result = ""
                status = "Copying models..."

                executor.execute {

                    try {

                        val modelDir =
                            File(context.filesDir, "sana_models")

                        if (!modelDir.exists()) {
                            modelDir.mkdirs()
                        }

                        val transformerFile =
                            File(modelDir, "sana_transformer.mnn")

                        val vaeFile =
                            File(modelDir, "sana_vae_decoder.mnn")

                        copyUriToFile(
                            context = context,
                            uri = transformer,
                            destination = transformerFile
                        )

                        mainHandler.post {
                            status = "Transformer copied"
                        }

                        copyUriToFile(
                            context = context,
                            uri = vae,
                            destination = vaeFile
                        )

                        mainHandler.post {
                            copying = false
                            status = "Models copied. Testing MNN..."
                        }

                        val output =
                            NativeSana.testModels(
                                context = context,
                                transformerFile = transformerFile,
                                vaeFile = vaeFile,
                                preferOpenCl = true
                            )

                        mainHandler.post {
                            result = output
                            status = "Test finished"
                            testing = false
                        }

                    } catch (t: Throwable) {

                        val message =
                            buildString {
                                append(t::class.java.simpleName)
                                append(": ")
                                append(t.message ?: "Unknown error")
                            }

                        mainHandler.post {
                            copying = false
                            testing = false
                            status = "Test failed"
                            result = message
                        }
                    }
                }
            },
            modifier = Modifier.fillMaxWidth()
        ) {

            if (testing) {

                CircularProgressIndicator(
                    modifier = Modifier
                        .width(22.dp)
                        .height(22.dp)
                )

                Spacer(modifier = Modifier.width(10.dp))

                Text("Testing...")

            } else {

                Text("Test Sana Models")
            }
        }

        Card(modifier = Modifier.fillMaxWidth()) {

            Column(
                modifier = Modifier.padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {

                Text(
                    text = "Status",
                    fontWeight = FontWeight.Bold
                )

                Text(text = status)

                if (result.isNotBlank()) {

                    Spacer(modifier = Modifier.height(4.dp))

                    Text(
                        text = "Result",
                        fontWeight = FontWeight.Bold
                    )

                    Text(text = result)
                }
            }
        }

        Spacer(modifier = Modifier.height(12.dp))

        Text(
            text = "This test loads the 1.5 GB Sana package from internal storage. The APK does not contain the models.",
            style = MaterialTheme.typography.bodySmall
        )
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

            destination.outputStream().use { output ->

                val buffer = ByteArray(1024 * 1024)

                while (true) {

                    val read = input.read(buffer)

                    if (read <= 0) break

                    output.write(buffer, 0, read)
                }

                output.flush()
            }
        }
        ?: throw IllegalStateException("Unable to open selected model")
}
