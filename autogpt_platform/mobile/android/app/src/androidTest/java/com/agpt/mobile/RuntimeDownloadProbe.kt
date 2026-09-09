package com.agpt.mobile

import android.annotation.SuppressLint
import android.content.Context
import android.content.Intent
import android.webkit.WebView

class RuntimeDownloadProbe(private val context: Context, private val runtime: RuntimeSupport) {
    @SuppressLint("SetJavaScriptEnabled")
    fun run() {
        val origin = requireNotNull(ServerOrigin.parse("https://android-runtime.invalid", false))
        val view = runtime.main { WebView(context).apply { settings.javaScriptEnabled = true } }
        var pickerRequests = 0
        lateinit var downloads: NativeDownloads
        runtime.main {
            downloads =
                NativeDownloads(context, view, origin) { intent ->
                    check(intent.action == Intent.ACTION_CREATE_DOCUMENT)
                    check(intent.getStringExtra(Intent.EXTRA_TITLE) == "runtime-probe.txt")
                    pickerRequests++
                    downloads.onDocumentCreated(null)
                }
            downloads.attach()
        }
        try {
            runtime.load(view, origin.chatUrl)
            check(runtime.main { origin.contains(view.url.orEmpty()) }) {
                "Offscreen document URL is ${runtime.main { view.url }}"
            }
            check(runtime.script(view, "typeof AutoGPTDownloads") == "\"object\"")
            runtime.script(
                view,
                """
                window.probeReply = null;
                AutoGPTDownloads.onmessage = event => { window.probeReply = JSON.parse(event.data).type; };
                AutoGPTDownloads.postMessage(JSON.stringify({type:'start', id:'top', filename:'runtime-probe.txt', mimeType:'text/plain', size:1}));
                """
                    .trimIndent(),
            )
            runtime.awaitScript(view, "window.probeReply", "\"cancelled\"")
            check(runtime.main { pickerRequests } == 1)
            runtime.script(
                view,
                """
                window.probeReply = null;
                AutoGPTDownloads.postMessage(JSON.stringify({type:'start', id:'bad', filename:'runtime-probe.txt', mimeType:'text/plain', size:-1}));
                """
                    .trimIndent(),
            )
            runtime.awaitScript(view, "window.probeReply", "\"error\"")
            check(runtime.main { pickerRequests } == 1)
            runtime.script(
                view,
                """
                window.frameDone = false;
                window.addEventListener('message', event => { if (event.data === 'frame-sent') window.frameDone = true; });
                const frame = document.createElement('iframe');
                frame.srcdoc = `<script>AutoGPTDownloads.postMessage(JSON.stringify({type:'start',id:'frame',filename:'runtime-probe.txt',mimeType:'text/plain',size:1})); parent.postMessage('frame-sent','*');<\/script>`;
                document.body.appendChild(frame);
                """
                    .trimIndent(),
            )
            runtime.awaitScript(view, "window.frameDone", "true")
            runtime.main { Unit }
            check(runtime.main { pickerRequests } == 1) {
                "A same-origin iframe reached the native picker"
            }
            runtime.load(view, "https://untrusted-runtime.invalid/")
            check(runtime.script(view, "typeof AutoGPTDownloads") == "\"undefined\"")
            check(runtime.main { pickerRequests } == 1)
        } finally {
            runtime.main {
                downloads.close()
                view.destroy()
            }
        }
    }
}
