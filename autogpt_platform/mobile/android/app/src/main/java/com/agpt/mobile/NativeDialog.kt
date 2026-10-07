package com.agpt.mobile

import android.app.Dialog
import android.content.Context
import android.graphics.Color
import android.os.Bundle
import android.view.View
import android.view.Window
import android.view.WindowManager
import android.widget.LinearLayout
import android.widget.ScrollView
import androidx.core.graphics.drawable.toDrawable

class NativeDialog(
    context: Context,
    title: String,
    detail: String,
    custom: View? = null,
    positive: String,
    onPositive: (NativeDialog) -> Unit,
) : Dialog(context) {
    init {
        requestWindowFeature(Window.FEATURE_NO_TITLE)
        val panel =
            LinearLayout(context).apply {
                orientation = LinearLayout.VERTICAL
                setPadding(dp(24), dp(28), dp(24), dp(24))
                background = NativeStyle.shape(context, context.getColor(R.color.shell_panel), 16)
                isFocusableInTouchMode = true
            }
        panel.addView(
            NativeStyle.text(context, title = true).apply {
                text = title
                textSize = 22f
                setLineHeight(NativeStyle.sp(context, 30))
            },
            LinearLayout.LayoutParams(-1, -2),
        )
        panel.addView(
            NativeStyle.text(context, muted = true).apply { text = detail },
            LinearLayout.LayoutParams(-1, -2).apply { topMargin = dp(12) },
        )
        custom?.let {
            panel.addView(it, LinearLayout.LayoutParams(-1, -2).apply { topMargin = dp(24) })
        }
        panel.addView(
            NativeStyle.button(context).apply {
                text = positive
                setOnClickListener { onPositive(this@NativeDialog) }
            },
            LinearLayout.LayoutParams(-1, -2).apply { topMargin = dp(28) },
        )
        panel.addView(
            NativeStyle.button(context, false).apply {
                setText(R.string.cancel)
                setOnClickListener { dismiss() }
            },
            LinearLayout.LayoutParams(-1, -2).apply { topMargin = dp(12) },
        )
        setContentView(
            ScrollView(context).apply {
                isVerticalScrollBarEnabled = false
                addView(panel)
            }
        )
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window?.setBackgroundDrawable(Color.TRANSPARENT.toDrawable())
        window?.setSoftInputMode(
            WindowManager.LayoutParams.SOFT_INPUT_ADJUST_RESIZE or
                WindowManager.LayoutParams.SOFT_INPUT_STATE_ALWAYS_HIDDEN
        )
    }

    override fun onStart() {
        super.onStart()
        window?.setLayout(
            minOf(context.resources.displayMetrics.widthPixels - dp(48), dp(464)),
            WindowManager.LayoutParams.WRAP_CONTENT,
        )
    }

    private fun dp(value: Int): Int = NativeStyle.dp(context, value)
}
