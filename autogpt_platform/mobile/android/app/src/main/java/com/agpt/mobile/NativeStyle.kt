package com.agpt.mobile

import android.content.Context
import android.content.res.ColorStateList
import android.graphics.Color
import android.graphics.drawable.GradientDrawable
import android.graphics.drawable.RippleDrawable
import android.graphics.drawable.StateListDrawable
import android.util.TypedValue
import android.widget.Button
import android.widget.TextView
import androidx.core.view.ViewCompat

object NativeStyle {
    fun dp(context: Context, value: Int): Int =
        (value * context.resources.displayMetrics.density).toInt()

    fun sp(context: Context, value: Int): Int =
        TypedValue.applyDimension(
                TypedValue.COMPLEX_UNIT_SP,
                value.toFloat(),
                context.resources.displayMetrics,
            )
            .toInt()

    fun text(context: Context, title: Boolean = false, muted: Boolean = false): TextView =
        TextView(context).apply {
            typeface =
                context.resources.getFont(
                    if (title) R.font.poppins_medium else R.font.geist_regular
                )
            textSize = if (title) 28f else 14f
            setLineHeight(sp(context, if (title) 40 else 22))
            includeFontPadding = false
            setTextColor(
                context.getColor(if (muted) R.color.shell_secondary else R.color.shell_primary)
            )
            if (title) {
                letterSpacing = -0.0075f
                ViewCompat.setAccessibilityHeading(this, true)
            }
        }

    fun shape(context: Context, color: Int, radius: Int, border: Int? = null): GradientDrawable =
        GradientDrawable().apply {
            setColor(color)
            cornerRadius = dp(context, radius).toFloat()
            border?.let { setStroke(dp(context, 1), it) }
        }

    fun button(context: Context, primary: Boolean = true): Button =
        Button(context).apply {
            isAllCaps = false
            typeface = context.resources.getFont(R.font.geist_medium)
            textSize = 14f
            setLineHeight(sp(context, 22))
            includeFontPadding = false
            minWidth = 0
            minimumWidth = 0
            minHeight = dp(context, 52)
            minimumHeight = dp(context, 52)
            setPadding(dp(context, 16), dp(context, 12), dp(context, 16), dp(context, 12))
            stateListAnimator = null
            elevation = 0f
            val surface =
                context.getColor(
                    if (primary) R.color.shell_accent else R.color.shell_secondary_button
                )
            val foreground =
                context.getColor(if (primary) R.color.shell_on_accent else R.color.shell_primary)
            val states =
                StateListDrawable().apply {
                    addState(
                        intArrayOf(-android.R.attr.state_enabled),
                        shape(
                            context,
                            context.getColor(
                                if (primary) R.color.shell_border else R.color.shell_faint
                            ),
                            999,
                        ),
                    )
                    addState(
                        intArrayOf(android.R.attr.state_pressed),
                        shape(
                            context,
                            context.getColor(
                                if (primary) R.color.shell_pressed else R.color.shell_border
                            ),
                            999,
                        ),
                    )
                    addState(intArrayOf(), shape(context, surface, 999))
                }
            background =
                RippleDrawable(
                    ColorStateList.valueOf((foreground and 0x00ffffff) or 0x22000000),
                    states,
                    null,
                )
            backgroundTintList = null
            setTextColor(
                ColorStateList(
                    arrayOf(intArrayOf(-android.R.attr.state_enabled), intArrayOf()),
                    intArrayOf(
                        context.getColor(
                            if (primary) R.color.shell_on_accent else R.color.shell_subtle
                        ),
                        foreground,
                    ),
                )
            )
        }

    fun iconBackground(context: Context) =
        RippleDrawable(
            ColorStateList.valueOf(0x18000000),
            shape(context, Color.TRANSPARENT, 999),
            shape(context, Color.WHITE, 999),
        )
}
