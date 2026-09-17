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
                    SanaModelTestScreen(
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
private fun SanaModelTestScreen(
    context: Context
) {

    var transformerUri by remember {
        mutableStateOf<Uri?>(null)
    }

    var vaeUri by remember {
        mutableStateOf<Uri?>(null)
    }

    var transformerName by remember {
        mutableStateOf("No Transformer selected")
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

    /*
     * ---------------------------------------------------------
     * TRANSFORMER FILE PICKER
     * ---------------------------------------------------------
     */

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

    /*
     * ---------------------------------------------------------
     * VAE FILE PICKER
     * ---------------------------------------------------------
     */

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

        /*
         * -----------------------------------------------------
         * HEADER
         * -----------------------------------------------------
         */

        Text(
            text = "Sana Android",
            style = MaterialTheme.typography.headlineMedium,
            fontWeight = FontWeight.Bold
        )

        Text(
            text = "Sana 0.6B • 512×512 • MNN • ARM64",
            style = MaterialTheme.typography.bodyMedium
        )

        Text(
            text = "Transformer + VAE diagnostic",
            style = MaterialTheme.typography.bodySmall
        )

        Spacer(
            modifier = Modifier.height(8.dp)
        )

        /*
         * -----------------------------------------------------
         * TRANSFORMER CARD
         * -----------------------------------------------------
         */

        Card(
            modifier = Modifier.fillMaxWidth()
        ) {

            Column(
                modifier = Modifier.padding(16.dp),

                verticalArrangement =
                    Arrangement.spacedBy(8.dp)
            ) {

                Text(
                    text = "1. Sana Transformer",
                    fontWeight = FontWeight.Bold
                )

                Text(
                    text = transformerName,
                    style = MaterialTheme.typography.bodySmall
                )

                Text(
                    text = "Transformer diagnostic was already passed.",
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

                    modifier =
                        Modifier.fillMaxWidth()
                ) {

                    Text(
                        text = "SELECT TRANSFORMER"
                    )
                }
            }
        }

        /*
         * -----------------------------------------------------
         * VAE CARD
         * -----------------------------------------------------
         */

        Card(
            modifier = Modifier.fillMaxWidth()
        ) {

            Column(
                modifier = Modifier.padding(16.dp),

                verticalArrangement =
                    Arrangement.spacedBy(8.dp)
            ) {

                Text(
                    text = "2. Sana VAE Decoder",
                    fontWeight = FontWeight.Bold
                )

                Text(
                    text = vaeName,
                    style = MaterialTheme.typography.bodySmall
                )

                Text(
                    text = "VAE zero-latent diagnostic passed.",
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

                    modifier =
                        Modifier.fillMaxWidth()
                ) {

                    Text(
                        text = "SELECT VAE"
                    )
                }
            }
        }

        /*
         * -----------------------------------------------------
         * SELECTED MODEL STATUS
         * -----------------------------------------------------
         */

        Card(
            modifier = Modifier.fillMaxWidth()
        ) {

            Column(
                modifier = Modifier.padding(16.dp),

                verticalArrangement =
                    Arrangement.spacedBy(6.dp)
            ) {

                Text(
                    text = "Selected models",
                    fontWeight = FontWeight.Bold
                )

                Text(
                    text =
                        if (transformerUri != null)
                            "✓ Transformer selected"
                        else
                            "○ Transformer not selected"
                )

                Text(
                    text =
                        if (vaeUri != null)
                            "✓ VAE selected"
                        else
                            "○ VAE not selected"
                )
            }
        }

        /*
         * -----------------------------------------------------
         * TEST BUTTON
         * -----------------------------------------------------
         */

        Button(
            enabled =
                transformerUri != null &&
                vaeUri != null &&
                !testing &&
                !copying,

            onClick = {

                val selectedTransformer =
                    transformerUri
                        ?: return@Button

                val selectedVae =
                    vaeUri
                        ?: return@Button

                testing = true
                copying = true

                result = ""

                status =
                    "Copying models..."

                executor.execute {

                    try {

                        /*
                         * -------------------------------------
                         * MODEL DIRECTORY
                         * -------------------------------------
                         */

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

                        /*
                         * -------------------------------------
                         * TRANSFORMER FILE
                         * -------------------------------------
                         */

                        val transformerFile =
                            File(
                                modelDirectory,
                                "sana_transformer.mnn"
                            )

                        copyUriToFile(
                            context = context,
                            uri = selectedTransformer,
                            destination = transformerFile
                        )

                        if (
                            transformerFile.length() <= 0L
                        ) {

                            throw IllegalStateException(
                                "Copied Transformer is empty"
                            )
                        }

                        mainHandler.post {

                            status =
                                "Transformer copied"
                        }

                        /*
                         * -------------------------------------
                         * VAE FILE
                         * -------------------------------------
                         */

                        val vaeFile =
                            File(
                                modelDirectory,
                                "sana_vae_decoder.mnn"
                            )

                        copyUriToFile(
                            context = context,
                            uri = selectedVae,
                            destination = vaeFile
                        )

                        if (
                            vaeFile.length() <= 0L
                        ) {

                            throw IllegalStateException(
                                "Copied VAE is empty"
                            )
                        }

                        mainHandler.post {

                            copying = false

                            status =
                                "Models copied. Starting native MNN test..."
                        }

                        /*
                         * -------------------------------------
                         * IMPORTANT
                         * -------------------------------------
                         *
                         * This restores the combined diagnostic.
                         *
                         * It passes BOTH:
                         *
                         * 1. Transformer
                         * 2. VAE
                         *
                         * to the current NativeSana API.
                         *
                         * This is NOT the old VAE-only test.
                         */

                        val output =
                            NativeSana.testModels(
                                context = context,
                                transformerFile =
                                    transformerFile,
                                vaeFile =
                                    vaeFile,
                                preferOpenCl = true
                            )

                        mainHandler.post {

                            result = output

                            status =
                                "Sana model test finished"

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

                                append("\n\n")

                                append(
                                    t.stackTraceToString()
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

                Row(
                    horizontalArrangement =
                        Arrangement.Center
                ) {

                    CircularProgressIndicator(
                        modifier =
                            Modifier
                                .width(22.dp)
                                .height(22.dp)
                    )

                    Spacer(
                        modifier =
                            Modifier.width(10.dp)
                    )

                    Text(
                        text = "TESTING..."
                    )
                }

            } else {

                Text(
                    text = "TEST TRANSFORMER + VAE"
                )
            }
        }

        /*
         * -----------------------------------------------------
         * STATUS
         * -----------------------------------------------------
         */

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

        /*
         * -----------------------------------------------------
         * DIAGNOSTIC INFORMATION
         * -----------------------------------------------------
         */

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
                    text = "Diagnostic mode",
                    fontWeight =
                        FontWeight.Bold
                )

                Text(
                    text =
                        "Transformer: previously validated"
                )

                Text(
                    text =
                        "VAE: zero-latent test validated"
                )

                Text(
                    text =
                        "Current screen tests both MNN models."
                )

                Text(
                    text =
                        "This is still a model diagnostic, not the final text-to-image pipeline."
                )
            }
        }
    }
}

/*
 * -------------------------------------------------------------
 * COPY MODEL FROM ANDROID DOCUMENT PROVIDER
 * -------------------------------------------------------------
 */

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
                        ByteArray(
                            1024 * 1024
                        )

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
            "Unable to open selected model"
        )
}
