package com.sana.android

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.os.ParcelFileDescriptor
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

    val executor = remember {
        Executors.newSingleThreadExecutor()
    }

    DisposableEffect(Unit) {
        onDispose {
            executor.shutdownNow()
        }
    }

    /*
     * ---------------------------------------------------------
     * TRANSFORMER PICKER
     * ---------------------------------------------------------
     */

    val transformerPicker =
        rememberLauncherForActivityResult(
            contract = ActivityResultContracts.OpenDocument()
        ) { uri ->

            if (uri != null) {

                try {
                    persistReadPermission(
                        context = context,
                        uri = uri
                    )
                } catch (_: Throwable) {
                }

                transformerUri = uri

                transformerName =
                    getDisplayName(
                        context,
                        uri
                    )

                status = "Transformer selected"
                result = ""
            }
        }

    /*
     * ---------------------------------------------------------
     * VAE PICKER
     * ---------------------------------------------------------
     */

    val vaePicker =
        rememberLauncherForActivityResult(
            contract = ActivityResultContracts.OpenDocument()
        ) { uri ->

            if (uri != null) {

                try {
                    persistReadPermission(
                        context = context,
                        uri = uri
                    )
                } catch (_: Throwable) {
                }

                vaeUri = uri

                vaeName =
                    getDisplayName(
                        context,
                        uri
                    )

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
            text = "Sana 0.6B • 512×512",
            style = MaterialTheme.typography.bodyMedium
        )

        Text(
            text = "MNN • ARM64 • External model loading",
            style = MaterialTheme.typography.bodySmall
        )

        Spacer(
            modifier = Modifier.height(8.dp)
        )

        /*
         * -----------------------------------------------------
         * IMPORTANT DESIGN NOTE
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
                    text = "No model copying",
                    fontWeight = FontWeight.Bold
                )

                Text(
                    text =
                        "The selected .mnn files are opened directly from device storage."
                )

                Text(
                    text =
                        "The app does not copy the 1+ GB Transformer into internal storage."
                )
            }
        }

        /*
         * -----------------------------------------------------
         * TRANSFORMER
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

                OutlinedButton(
                    enabled = !testing,

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
         * VAE
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

                OutlinedButton(
                    enabled = !testing,

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
         * MODEL STATUS
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
                    text = "Models",
                    fontWeight = FontWeight.Bold
                )

                Text(
                    text =
                        if (transformerUri != null)
                            "✓ Transformer ready"
                        else
                            "○ Transformer not selected"
                )

                Text(
                    text =
                        if (vaeUri != null)
                            "✓ VAE ready"
                        else
                            "○ VAE not selected"
                )
            }
        }

        /*
         * -----------------------------------------------------
         * TEST BOTH
         * -----------------------------------------------------
         */

        Button(
            enabled =
                transformerUri != null &&
                vaeUri != null &&
                !testing,

            onClick = {

                val transformer =
                    transformerUri
                        ?: return@Button

                val vae =
                    vaeUri
                        ?: return@Button

                testing = true
                result = ""

                status =
                    "Opening models..."

                executor.execute {

                    var transformerFd: ParcelFileDescriptor? =
                        null

                    var vaeFd: ParcelFileDescriptor? =
                        null

                    try {

                        /*
                         * -------------------------------------
                         * OPEN TRANSFORMER
                         * -------------------------------------
                         */

                        transformerFd =
                            context.contentResolver
                                .openFileDescriptor(
                                    transformer,
                                    "r"
                                )

                            ?: throw IllegalStateException(
                                "Unable to open Transformer"
                            )

                        /*
                         * -------------------------------------
                         * OPEN VAE
                         * -------------------------------------
                         */

                        vaeFd =
                            context.contentResolver
                                .openFileDescriptor(
                                    vae,
                                    "r"
                                )

                            ?: throw IllegalStateException(
                                "Unable to open VAE"
                            )

                        /*
                         * Do NOT close these descriptors before
                         * native testing finishes.
                         */

                        val transformerDescriptor =
                            transformerFd
                                .detachFd()

                        transformerFd = null

                        val vaeDescriptor =
                            vaeFd
                                .detachFd()

                        vaeFd = null

                        runOnUiThread {

                            status =
                                "Running Transformer + VAE diagnostics..."
                        }

                        /*
                         * -------------------------------------
                         * NATIVE TEST
                         * -------------------------------------
                         *
                         * Native code owns the detached FDs
                         * and closes them after MNN finishes.
                         */

                        val nativeResult =
                            try {

                                NativeSana.testModels(
                                    transformerFd =
                                        transformerDescriptor,

                                    vaeFd =
                                        vaeDescriptor,

                                    preferOpenCl = false
                                )

                            } finally {

                                /*
                                 * Native method closes both
                                 * descriptors.
                                 */
                            }

                        runOnUiThread {

                            result =
                                nativeResult

                            status =
                                "Model test finished"

                            testing = false
                        }

                    } catch (t: Throwable) {

                        try {
                            transformerFd?.close()
                        } catch (_: Throwable) {
                        }

                        try {
                            vaeFd?.close()
                        } catch (_: Throwable) {
                        }

                        val error =
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

                        runOnUiThread {

                            status =
                                "Model test failed"

                            result =
                                error

                            testing = false
                        }
                    }
                }
            },

            modifier =
                Modifier.fillMaxWidth()
        ) {

            if (testing) {

                CircularProgressIndicator(
                    modifier =
                        Modifier
                            .height(22.dp)
                )

            } else {

                Text(
                    text =
                        "TEST TRANSFORMER + VAE"
                )
            }
        }

        /*
         * -----------------------------------------------------
         * RESULT
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
                    fontWeight =
                        FontWeight.Bold
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
                        "Transformer test: CPU"
                )

                Text(
                    text =
                        "VAE test: CPU"
                )

                Text(
                    text =
                        "OpenCL is intentionally disabled for this diagnostic build."
                )

                Text(
                    text =
                        "The models remain in their original device location."
                )
            }
        }
    }
}

/*
 * -------------------------------------------------------------
 * PERSIST URI READ ACCESS
 * -------------------------------------------------------------
 */

private fun persistReadPermission(
    context: Context,
    uri: Uri
) {

    val flags =
        Intent.FLAG_GRANT_READ_URI_PERMISSION

    try {

        context.contentResolver
            .takePersistableUriPermission(
                uri,
                flags
            )

    } catch (_: SecurityException) {

        /*
         * Some providers do not offer persistable
         * permissions. The temporary picker grant
         * is still valid during the current operation.
         */
    }
}

/*
 * -------------------------------------------------------------
 * DISPLAY NAME
 * -------------------------------------------------------------
 */

private fun getDisplayName(
    context: Context,
    uri: Uri
): String {

    var name: String? = null

    try {

        context.contentResolver
            .query(
                uri,
                arrayOf(
                    "_display_name"
                ),
                null,
                null,
                null
            )
            ?.use { cursor ->

                if (cursor.moveToFirst()) {

                    val index =
                        cursor.getColumnIndex(
                            "_display_name"
                        )

                    if (index >= 0) {

                        name =
                            cursor.getString(index)
                    }
                }
            }

    } catch (_: Throwable) {
    }

    return name
        ?: uri.lastPathSegment
        ?: "Selected model"
}
