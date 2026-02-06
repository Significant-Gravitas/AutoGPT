require.config({ paths: { 'vs': 'assets/monaco-editor/min/vs' } });
require(['vs/editor/editor.main'], function() {
    // Monaco is loaded, now we can load our custom language and IDE logic
    var script = document.createElement('script');
    script.src = 'deluge-lang.js';
    script.onload = function() {
        var ideScript = document.createElement('script');
        ideScript.src = 'ide.js';
        document.body.appendChild(ideScript);
    };
    document.body.appendChild(script);
});
