package com.agpt.mobile

import android.content.Context
import android.view.View
import android.view.inputmethod.EditorInfo
import android.widget.EditText
import android.widget.LinearLayout

object ServerSettingsDialog {
    fun create(
        context: Context,
        origin: ServerOrigin,
        onConnect: (ServerOrigin) -> Unit,
    ): NativeDialog {
        val input =
            EditText(context).apply {
                id = View.generateViewId()
                setText(origin.value)
                hint = ServerOrigin.DEFAULT
                inputType = EditorInfo.TYPE_CLASS_TEXT or EditorInfo.TYPE_TEXT_VARIATION_URI
                setSingleLine()
                typeface = resources.getFont(R.font.geist_regular)
                textSize = 14f
                setTextColor(context.getColor(R.color.shell_primary))
                setHintTextColor(context.getColor(R.color.shell_muted))
                minHeight = NativeStyle.dp(context, 46)
                setPadding(
                    NativeStyle.dp(context, 16),
                    NativeStyle.dp(context, 12),
                    NativeStyle.dp(context, 16),
                    NativeStyle.dp(context, 12),
                )
                background =
                    NativeStyle.shape(
                        context,
                        context.getColor(R.color.shell_panel),
                        12,
                        context.getColor(R.color.shell_border),
                    )
                backgroundTintList = null
                setOnFocusChangeListener { _, focused ->
                    background =
                        NativeStyle.shape(
                            context,
                            context.getColor(R.color.shell_panel),
                            12,
                            context.getColor(
                                if (focused) R.color.shell_focus else R.color.shell_border
                            ),
                        )
                }
                selectAll()
            }
        val error =
            NativeStyle.text(context).apply {
                textSize = 12f
                setTextColor(context.getColor(R.color.shell_error))
                visibility = View.GONE
                accessibilityLiveRegion = View.ACCESSIBILITY_LIVE_REGION_POLITE
            }
        val form =
            LinearLayout(context).apply {
                orientation = LinearLayout.VERTICAL
                addView(
                    NativeStyle.text(context).apply {
                        setText(R.string.server_address)
                        labelFor = input.id
                        typeface = resources.getFont(R.font.geist_medium)
                    },
                    LinearLayout.LayoutParams(-1, -2),
                )
                addView(
                    input,
                    LinearLayout.LayoutParams(-1, -2).apply {
                        topMargin = NativeStyle.dp(context, 8)
                    },
                )
                addView(
                    error,
                    LinearLayout.LayoutParams(-1, -2).apply {
                        topMargin = NativeStyle.dp(context, 8)
                    },
                )
                if (BuildConfig.DEBUG)
                    addView(
                        NativeStyle.text(context, muted = true).apply {
                            setText(R.string.server_debug_explanation)
                            textSize = 12f
                        },
                        LinearLayout.LayoutParams(-1, -2).apply {
                            topMargin = NativeStyle.dp(context, 12)
                        },
                    )
            }
        return NativeDialog(
            context,
            context.getString(R.string.settings),
            context.getString(R.string.server_settings_detail),
            form,
            context.getString(R.string.save),
        ) { dialog ->
            val candidate = ServerOrigin.parse(input.text.toString(), BuildConfig.DEBUG)
            if (candidate == null) {
                error.setText(R.string.server_invalid)
                error.visibility = View.VISIBLE
            } else if (candidate == origin) dialog.dismiss()
            else
                NativeDialog(
                        context,
                        context.getString(R.string.switch_server_title),
                        context.getString(R.string.switch_server_detail, candidate.value),
                        positive = context.getString(R.string.connect),
                    ) { confirmation ->
                        confirmation.dismiss()
                        dialog.dismiss()
                        onConnect(candidate)
                    }
                    .show()
        }
    }
}
