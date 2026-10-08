package com.agpt.mobile

import android.app.Instrumentation
import android.content.Context
import android.content.Intent
import android.view.View
import android.view.ViewGroup
import android.webkit.WebView

class RuntimePushProbe(
    private val instrumentation: Instrumentation,
    private val runtime: RuntimeSupport,
) {
    fun run() {
        val context = instrumentation.targetContext
        val origin = "https://push-runtime.invalid"
        runtime.main {
            context
                .getSharedPreferences("server", Context.MODE_PRIVATE)
                .edit()
                .putString("origin", origin)
                .commit()
        }
        val activity =
            instrumentation.startActivitySync(
                Intent(context, MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            )
        try {
            val view = runtime.main { requireNotNull(findWebView(activity.window.decorView)) }
            runtime.load(view, origin)
            runtime.script(
                view,
                "window.pushResult=null; AutoGPTPush.onmessage=e=>window.pushResult=JSON.parse(e.data).permission; AutoGPTPush.postMessage(JSON.stringify({id:'12345678-1234-4234-8234-123456789abc',action:'status',account_id:'fixture'}))",
            )
            runtime.awaitScript(view, "window.pushResult", "\"disabled\"")
            runtime.script(
                view,
                "window.frameReply=false; window.addEventListener('message',e=>{if(e.data==='reply')window.frameReply=true}); const f=document.createElement('iframe'); f.srcdoc=\"<script>if(typeof AutoGPTPush!=='undefined'){AutoGPTPush.onmessage=()=>parent.postMessage('reply','*');AutoGPTPush.postMessage(JSON.stringify({id:'12345678-1234-4234-8234-123456789abc',action:'enable',account_id:'fixture'}));}parent.postMessage('attempted','*')<\\/script>\"; window.frameAttempted=false; window.addEventListener('message',e=>{if(e.data==='attempted')window.frameAttempted=true}); document.body.appendChild(f)",
            )
            runtime.awaitScript(view, "window.frameAttempted", "true")
            Thread.sleep(500)
            check(runtime.script(view, "window.frameReply") == "false")
            runtime.load(view, "https://untrusted-runtime.invalid")
            check(runtime.script(view, "typeof AutoGPTPush") == "\"undefined\"")
        } finally {
            runtime.main {
                activity.finish()
                context
                    .getSharedPreferences("server", Context.MODE_PRIVATE)
                    .edit()
                    .remove("origin")
                    .commit()
            }
        }
    }

    private fun findWebView(view: View): WebView? {
        if (view is WebView) return view
        if (view is ViewGroup)
            for (index in 0 until view.childCount) {
                findWebView(view.getChildAt(index))?.let {
                    return it
                }
            }
        return null
    }
}
