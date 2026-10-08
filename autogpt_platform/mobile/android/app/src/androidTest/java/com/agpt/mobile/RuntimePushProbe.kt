package com.agpt.mobile

import android.app.Instrumentation
import android.content.Intent

class RuntimePushProbe(
    private val instrumentation: Instrumentation,
    private val runtime: RuntimeSupport,
) {
    fun run() {
        val context = instrumentation.targetContext
        val origin = "https://push-runtime.invalid"
        val activity =
            instrumentation.startActivitySync(
                Intent(context, PushProbeActivity::class.java)
                    .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            ) as PushProbeActivity
        var stage = "load trusted document"
        try {
            val view = runtime.main { activity.browser }
            runtime.load(view, origin)
            stage = "main-frame status"
            runtime.script(
                view,
                "window.pushResult=null; AutoGPTPush.onmessage=e=>window.pushResult=JSON.parse(e.data).permission; AutoGPTPush.postMessage(JSON.stringify({id:'12345678-1234-4234-8234-123456789abc',action:'status',account_id:'fixture'}))",
            )
            runtime.awaitScript(view, "window.pushResult", "\"disabled\"")
            stage = "child-frame rejection"
            runtime.script(
                view,
                "window.frameReply=false; window.addEventListener('message',e=>{if(e.data==='reply')window.frameReply=true}); const f=document.createElement('iframe'); f.srcdoc=\"<script>if(typeof AutoGPTPush!=='undefined'){AutoGPTPush.onmessage=()=>parent.postMessage('reply','*');AutoGPTPush.postMessage(JSON.stringify({id:'12345678-1234-4234-8234-123456789abc',action:'enable',account_id:'fixture'}));}parent.postMessage('attempted','*')<\\/script>\"; window.frameAttempted=false; window.addEventListener('message',e=>{if(e.data==='attempted')window.frameAttempted=true}); document.body.appendChild(f)",
            )
            runtime.awaitScript(view, "window.frameAttempted", "true")
            Thread.sleep(500)
            check(runtime.script(view, "window.frameReply") == "false")
            stage = "untrusted-origin rejection"
            runtime.load(view, "https://untrusted-runtime.invalid")
            check(runtime.script(view, "typeof AutoGPTPush") == "\"undefined\"")
        } catch (error: Exception) {
            throw IllegalStateException(
                "Push probe failed during $stage: ${error.javaClass.simpleName}",
                error,
            )
        } finally {
            runtime.main { activity.finish() }
        }
    }
}
