// MainActivity.kt (acts as home screen)
package com.example.facerecognizer

import android.content.Intent
import android.os.Bundle
import android.widget.EditText
import androidx.appcompat.app.AppCompatActivity
import com.example.facerecognizer.databinding.ActivityMainBinding
import androidx.appcompat.app.AlertDialog

class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.btnFaceReg.setOnClickListener {
                showNameIdDialog()
        }
    }

    private fun showNameIdDialog() {
        val dialogView = layoutInflater.inflate(R.layout.dialog_name_id, null)
        val nameInput = dialogView.findViewById<EditText>(R.id.etDialogName)
        val idInput = dialogView.findViewById<EditText>(R.id.etDialogId)

        AlertDialog.Builder(this)
            .setTitle("Enter Name and ID")
            .setView(dialogView)
            .setPositiveButton("Start") { _, _ ->
                val name = nameInput.text.toString().trim()
                val id = idInput.text.toString().trim()

                if (name.isNotEmpty() && id.isNotEmpty()) {
                    val intent = Intent(this, FaceRegisterActivity::class.java).apply {
                        putExtra("USER_NAME", name)
                        putExtra("USER_ID", id)
                    }
                    startActivity(intent)
                }
            }
            .setNegativeButton("Cancel", null)
            .show()
    }
}
