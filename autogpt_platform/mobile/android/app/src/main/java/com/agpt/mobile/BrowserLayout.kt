package com.agpt.mobile

import android.content.Context
import android.graphics.Typeface
import android.view.Gravity
import android.view.View
import android.widget.Button
import android.widget.FrameLayout
import android.widget.ImageButton
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.ScrollView
import android.widget.TextView

class BrowserLayout(context: Context) : LinearLayout(context) {
    val back = ImageButton(context)
    val menu = ImageButton(context)
    val webContainer = FrameLayout(context)
    private val content = FrameLayout(context)
    private val progress = ProgressBar(context, null, android.R.attr.progressBarStyleHorizontal)
    private val panel = LinearLayout(context)
    private val panelScroll = ScrollView(context)
    private val panelTitle = TextView(context)
    private val panelDetail = TextView(context)
    private val action = Button(context)
    private val secondaryAction = Button(context)

    init {
        orientation = VERTICAL
        setBackgroundColor(color(R.color.shell_background))
        val toolbar =
            LinearLayout(context).apply {
                gravity = Gravity.CENTER_VERTICAL
                setBackgroundColor(color(R.color.shell_background))
            }
        back.apply {
            setImageResource(R.drawable.ic_back)
            contentDescription = context.getString(R.string.back)
            setBackgroundColor(android.graphics.Color.TRANSPARENT)
        }
        toolbar.addView(back, LayoutParams(dp(48), dp(48)))
        val title =
            TextView(context).apply {
                text = context.getString(R.string.app_name)
                textSize = 18f
                setTextColor(color(R.color.shell_primary))
                setTypeface(typeface, Typeface.BOLD)
                gravity = Gravity.CENTER_VERTICAL
                importantForAccessibility = View.IMPORTANT_FOR_ACCESSIBILITY_YES
            }
        toolbar.addView(title, LayoutParams(0, dp(48), 1f))
        menu.apply {
            setImageResource(R.drawable.ic_more)
            contentDescription = context.getString(R.string.app_menu)
            setBackgroundColor(android.graphics.Color.TRANSPARENT)
        }
        toolbar.addView(menu, LayoutParams(dp(48), dp(48)))
        addView(toolbar, LayoutParams(LayoutParams.MATCH_PARENT, dp(48)))
        progress.apply {
            max = 100
            progressTintList =
                android.content.res.ColorStateList.valueOf(color(R.color.shell_accent))
            visibility = INVISIBLE
        }
        addView(progress, LayoutParams(LayoutParams.MATCH_PARENT, dp(2)))
        content.addView(
            webContainer,
            FrameLayout.LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.MATCH_PARENT),
        )
        panel.apply {
            orientation = VERTICAL
            gravity = Gravity.CENTER
            setPadding(dp(32), dp(24), dp(32), dp(24))
            setBackgroundColor(color(R.color.shell_panel))
        }
        panel.addView(
            ImageView(context).apply {
                setImageResource(R.drawable.ic_autogpt)
                importantForAccessibility = View.IMPORTANT_FOR_ACCESSIBILITY_NO
            },
            LayoutParams(dp(72), dp(72)).apply { bottomMargin = dp(24) },
        )
        panelTitle.apply {
            textSize = 24f
            setTextColor(color(R.color.shell_primary))
            setTypeface(typeface, Typeface.BOLD)
            gravity = Gravity.CENTER
            accessibilityLiveRegion = View.ACCESSIBILITY_LIVE_REGION_POLITE
        }
        panel.addView(
            panelTitle,
            LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.WRAP_CONTENT),
        )
        panelDetail.apply {
            textSize = 16f
            gravity = Gravity.CENTER
            setTextColor(color(R.color.shell_secondary))
            setLineSpacing(dp(3).toFloat(), 1f)
        }
        panel.addView(
            panelDetail,
            LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.WRAP_CONTENT).apply {
                topMargin = dp(12)
                bottomMargin = dp(24)
            },
        )
        action.isAllCaps = false
        secondaryAction.isAllCaps = false
        panel.addView(action, LayoutParams(LayoutParams.WRAP_CONTENT, LayoutParams.WRAP_CONTENT))
        panel.addView(
            secondaryAction,
            LayoutParams(LayoutParams.WRAP_CONTENT, LayoutParams.WRAP_CONTENT),
        )
        panelScroll.apply {
            isFillViewport = true
            setBackgroundColor(color(R.color.shell_panel))
            visibility = GONE
            addView(
                panel,
                FrameLayout.LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.WRAP_CONTENT),
            )
        }
        content.addView(
            panelScroll,
            FrameLayout.LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.MATCH_PARENT),
        )
        addView(content, LayoutParams(LayoutParams.MATCH_PARENT, 0, 1f))
    }

    fun loading(value: Int) {
        progress.progress = value
        progress.visibility = if (value in 0..99) VISIBLE else INVISIBLE
    }

    fun showPage() {
        panelScroll.visibility = GONE
        webContainer.importantForAccessibility = View.IMPORTANT_FOR_ACCESSIBILITY_AUTO
    }

    fun showPanel(
        title: Int,
        detail: Int,
        button: Int,
        onClick: () -> Unit,
        secondary: Pair<Int, () -> Unit>? = null,
    ) {
        loading(100)
        panelTitle.setText(title)
        panelDetail.setText(detail)
        action.visibility = if (button == 0) GONE else VISIBLE
        if (button != 0) action.setText(button)
        action.setOnClickListener { onClick() }
        secondaryAction.visibility = if (secondary == null) GONE else VISIBLE
        secondary?.let { (label, handler) ->
            secondaryAction.setText(label)
            secondaryAction.setOnClickListener { handler() }
        }
        panelScroll.visibility = VISIBLE
        panelScroll.scrollTo(0, 0)
        webContainer.importantForAccessibility =
            View.IMPORTANT_FOR_ACCESSIBILITY_NO_HIDE_DESCENDANTS
    }

    private fun color(id: Int): Int = context.getColor(id)

    private fun dp(value: Int): Int = (value * resources.displayMetrics.density).toInt()
}
