package com.sana.android

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Slider
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.runtime.remember
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp

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

    var prompt by remember {
        mutableStateOf("")
    }

    var steps by remember {
        mutableFloatStateOf(4f)
    }

    var fastMode by remember {
        mutableStateOf(true)
    }

    var generating by remember {
        mutableStateOf(false)
    }

    Surface(
        modifier = Modifier.fillMaxSize(),
        color = Color(0xFFF8F8FA)
    ) {

        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(20.dp),
            verticalArrangement = Arrangement.Top
        ) {

            Text(
                text = "Sana",
                style = MaterialTheme.typography.headlineLarge,
                fontWeight = FontWeight.Bold
            )

            Text(
                text = "On-device AI image generation",
                style = MaterialTheme.typography.bodyMedium,
                color = Color.Gray
            )

            Spacer(
                modifier = Modifier.height(24.dp)
            )

            OutlinedTextField(
                value = prompt,
                onValueChange = {
                    prompt = it
                },
                modifier = Modifier
                    .fillMaxWidth()
                    .height(150.dp),
                shape = RoundedCornerShape(18.dp),
                label = {
                    Text("Prompt")
                },
                placeholder = {
                    Text(
                        "Describe the image you want..."
                    )
                }
            )

            Spacer(
                modifier = Modifier.height(20.dp)
            )

            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween
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

            Spacer(
                modifier = Modifier.height(18.dp)
            )

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

            Spacer(
                modifier = Modifier.height(20.dp)
            )

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

            Spacer(
                modifier = Modifier.height(28.dp)
            )

            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(300.dp)
                    .background(
                        Color.White,
                        RoundedCornerShape(22.dp)
                    ),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.Center
            ) {

                Text(
                    text = if (generating)
                        "Preparing Sana engine..."
                    else
                        "Generated image will appear here",
                    color = Color.Gray
                )
            }
        }
    }
}
