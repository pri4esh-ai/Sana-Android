package com.sana.android

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.sana.android.engine.NativeSana

class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContent {
            SanaApp()
        }
    }
}

@Composable
private fun SanaApp() {

    var prompt by remember { mutableStateOf("") }
    var steps by remember { mutableFloatStateOf(4f) }
    var fastMode by remember { mutableStateOf(true) }
    var generating by remember { mutableStateOf(false) }

    /*
     * NativeSana no longer has isAvailable().
     *
     * The new native API exposes:
     * - isInitialized()
     * - backend()
     * - status()
     *
     * The actual Sana .mnn model will be initialized later,
     * after the model assets are added.
     */
    val nativeInitialized = remember {
        NativeSana.isInitialized()
    }

    val backend = remember {
        NativeSana.backend()
    }

    val status = remember {
        NativeSana.status()
    }

    Surface(
        modifier = Modifier.fillMaxSize(),
        color = Color(0xFFF8F8FA)
    ) {

        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(20.dp)
        ) {

            Text(
                text = "Sana",
                style = MaterialTheme.typography.headlineLarge,
                fontWeight = FontWeight.Bold
            )

            Text(
                text = "On-device AI image generation",
                color = Color.Gray
            )

            Spacer(modifier = Modifier.height(12.dp))

            Card(
                colors = CardDefaults.cardColors(
                    containerColor = Color.White
                )
            ) {

                Column(
                    modifier = Modifier.padding(16.dp)
                ) {

                    Text(
                        text = "Native Engine",
                        fontWeight = FontWeight.Bold
                    )

                    Spacer(modifier = Modifier.height(4.dp))

                    Text(
                        text = if (nativeInitialized)
                            "✓ Initialized"
                        else
                            "○ Waiting for model",
                        color = if (nativeInitialized)
                            Color(0xFF2E7D32)
                        else
                            Color.Gray
                    )

                    Text(
                        text = "Backend: $backend",
                        color = Color.Gray
                    )

                    Text(
                        text = "Status: $status",
                        color = Color.Gray
                    )
                }
            }

            Spacer(modifier = Modifier.height(20.dp))

            OutlinedTextField(
                value = prompt,
                onValueChange = { prompt = it },
                modifier = Modifier
                    .fillMaxWidth()
                    .height(150.dp),
                shape = RoundedCornerShape(18.dp),
                label = {
                    Text("Prompt")
                },
                placeholder = {
                    Text("Describe the image you want...")
                }
            )

            Spacer(modifier = Modifier.height(20.dp))

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {

                Column {

                    Text(
                        text = "Fast mode",
                        fontWeight = FontWeight.SemiBold
                    )

                    Text(
                        text = if (fastMode)
                            "Optimized generation"
                        else
                            "Higher quality",
                        color = Color.Gray
                    )
                }

                Switch(
                    checked = fastMode,
                    onCheckedChange = {
                        fastMode = it
                    }
                )
            }

            Spacer(modifier = Modifier.height(18.dp))

            Text(
                text = "Steps: ${steps.toInt()}",
                fontWeight = FontWeight.SemiBold
            )

            Slider(
                value = steps,
                onValueChange = {
                    steps = it
                },
                valueRange = 1f..8f,
                steps = 6
            )

            Spacer(modifier = Modifier.height(20.dp))

            Button(
                onClick = {
                    if (prompt.isNotBlank()) {
                        generating = true
                    }
                },
                enabled = prompt.isNotBlank() && !generating,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(56.dp),
                shape = RoundedCornerShape(18.dp)
            ) {

                if (generating) {

                    CircularProgressIndicator(
                        modifier = Modifier.size(22.dp),
                        strokeWidth = 2.dp
                    )

                } else {

                    Text(
                        text = "Generate",
                        fontWeight = FontWeight.Bold
                    )
                }
            }

            Spacer(modifier = Modifier.height(28.dp))

            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(300.dp)
                    .background(
                        color = Color.White,
                        shape = RoundedCornerShape(22.dp)
                    ),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.Center
            ) {

                Text(
                    text = if (generating)
                        "Sana generation pipeline will appear here"
                    else
                        "Generated image will appear here",
                    color = Color.Gray
                )
            }
        }
    }
}
