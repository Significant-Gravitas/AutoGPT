package com.agpt.mobile

import android.content.Context
import android.content.res.ColorStateList
import android.view.Gravity
import android.widget.FrameLayout
import android.widget.ImageButton
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.ScrollView
import androidx.core.view.isVisible

class BrowserLayout(context: Context) : LinearLayout(context) {
    val back = ImageButton(context)
    val menu = ImageButton(context)
    val webContainer = FrameLayout(context)
    private val progress = ProgressBar(context, null, android.R.attr.progressBarStyleHorizontal)
    private val panelScroll = ScrollView(context)
    private val panelTitle = NativeStyle.text(context, title = true)
    private val panelDetail = NativeStyle.text(context, muted = true)
    private val action = NativeStyle.button(context)
    private val secondaryAction = NativeStyle.button(context, primary = false)
    private val toolbarLogo = ImageView(context)
    private val spinner = ProgressBar(context)

    init {
        orientation = VERTICAL
        setBackgroundColor(context.getColor(R.color.shell_background))
        val toolbar = FrameLayout(context).apply { setPadding(dp(12), 0, dp(12), 0) }
        back.apply {
            setImageResource(R.drawable.ic_back)
            contentDescription = context.getString(R.string.back)
            background = NativeStyle.iconBackground(context)
        }
        menu.apply {
            setImageResource(R.drawable.ic_more)
            contentDescription = context.getString(R.string.app_menu)
            background = NativeStyle.iconBackground(context)
        }
        toolbarLogo.apply {
            setImageResource(R.drawable.autogpt_wordmark)
            contentDescription = context.getString(R.string.app_name)
            scaleType = ImageView.ScaleType.FIT_CENTER
        }
        toolbar.addView(
            back,
            FrameLayout.LayoutParams(dp(48), dp(48), Gravity.START or Gravity.CENTER_VERTICAL),
        )
        toolbar.addView(toolbarLogo, FrameLayout.LayoutParams(dp(89), dp(40), Gravity.CENTER))
        toolbar.addView(
            menu,
            FrameLayout.LayoutParams(dp(48), dp(48), Gravity.END or Gravity.CENTER_VERTICAL),
        )
        addView(toolbar, LayoutParams(LayoutParams.MATCH_PARENT, dp(56)))
        progress.apply {
            max = 100
            progressTintList = ColorStateList.valueOf(context.getColor(R.color.shell_accent))
            visibility = INVISIBLE
        }
        addView(progress, LayoutParams(LayoutParams.MATCH_PARENT, dp(2)))
        val content = FrameLayout(context)
        content.addView(
            webContainer,
            FrameLayout.LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.MATCH_PARENT),
        )
        val centered =
            LinearLayout(context).apply {
                orientation = VERTICAL
                gravity = Gravity.CENTER
                setPadding(dp(24), dp(40), dp(24), dp(48))
                setBackgroundColor(context.getColor(R.color.shell_panel))
            }
        val column =
            object : LinearLayout(context) {
                    override fun onMeasure(widthMeasureSpec: Int, heightMeasureSpec: Int) {
                        super.onMeasure(
                            MeasureSpec.makeMeasureSpec(
                                minOf(MeasureSpec.getSize(widthMeasureSpec), dp(416)),
                                MeasureSpec.getMode(widthMeasureSpec),
                            ),
                            heightMeasureSpec,
                        )
                    }
                }
                .apply { orientation = VERTICAL }
        column.addView(
            ImageView(context).apply {
                setImageResource(R.drawable.autogpt_wordmark)
                importantForAccessibility = IMPORTANT_FOR_ACCESSIBILITY_NO
                scaleType = ImageView.ScaleType.FIT_CENTER
            },
            LayoutParams(dp(128), dp(58)).apply {
                gravity = Gravity.CENTER_HORIZONTAL
                bottomMargin = dp(40)
            },
        )
        spinner.apply {
            indeterminateTintList = ColorStateList.valueOf(context.getColor(R.color.shell_accent))
            visibility = GONE
        }
        column.addView(spinner, LayoutParams(dp(24), dp(24)).apply { bottomMargin = dp(20) })
        panelTitle.accessibilityLiveRegion = ACCESSIBILITY_LIVE_REGION_POLITE
        column.addView(
            panelTitle,
            LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.WRAP_CONTENT),
        )
        column.addView(
            panelDetail,
            LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.WRAP_CONTENT).apply {
                topMargin = dp(12)
                bottomMargin = dp(32)
            },
        )
        column.addView(action, LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.WRAP_CONTENT))
        column.addView(
            secondaryAction,
            LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.WRAP_CONTENT).apply {
                topMargin = dp(12)
            },
        )
        centered.addView(column, LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.WRAP_CONTENT))
        panelScroll.apply {
            isFillViewport = true
            visibility = GONE
            isVerticalScrollBarEnabled = false
            addView(
                centered,
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
        spinner.isVisible = panelScroll.isVisible && value in 0..99
    }

    fun showLoading() {
        showPanel(R.string.loading_title, R.string.loading_detail, 0, {})
        loading(5)
    }

    fun showPage() {
        panelScroll.visibility = GONE
        toolbarLogo.visibility = VISIBLE
        back.visibility = VISIBLE
        webContainer.importantForAccessibility = IMPORTANT_FOR_ACCESSIBILITY_AUTO
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
        toolbarLogo.visibility = INVISIBLE
        back.visibility = if (back.isEnabled) VISIBLE else INVISIBLE
        panelScroll.visibility = VISIBLE
        panelScroll.scrollTo(0, 0)
        webContainer.importantForAccessibility = IMPORTANT_FOR_ACCESSIBILITY_NO_HIDE_DESCENDANTS
    }

    private fun dp(value: Int): Int = NativeStyle.dp(context, value)
}
