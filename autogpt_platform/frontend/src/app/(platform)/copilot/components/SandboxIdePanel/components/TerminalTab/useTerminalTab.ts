import { getWebSocketToken } from "@/lib/supabase/actions";
import { environment } from "@/services/environment";
import { FitAddon } from "@xterm/addon-fit";
import { Terminal } from "@xterm/xterm";
import { useEffect, useRef, useState } from "react";
import { buildTerminalWsUrl, xtermTheme } from "../../helpers";

export function useTerminalTab(sessionId: string) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [isClosed, setIsClosed] = useState(false);
  const [reconnectKey, setReconnectKey] = useState(0);

  useEffect(() => {
    let disposed = false;
    let ws: WebSocket | null = null;
    let term: Terminal | null = null;
    let fitAddon: FitAddon | null = null;
    let resizeObserver: ResizeObserver | null = null;

    function sendResize() {
      if (!term || !ws || ws.readyState !== WebSocket.OPEN) return;
      ws.send(
        JSON.stringify({ type: "resize", cols: term.cols, rows: term.rows }),
      );
    }

    async function start() {
      const element = containerRef.current;
      if (!element) return;

      term = new Terminal({
        theme: xtermTheme,
        fontSize: 12,
        fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
        cursorBlink: true,
        convertEol: true,
      });
      fitAddon = new FitAddon();
      term.loadAddon(fitAddon);
      term.open(element);
      fitAddon.fit();

      const { token, error } = await getWebSocketToken();
      if (disposed || !term) return;
      if (!token) {
        term.write(`\r\n[terminal] auth error: ${error ?? "no token"}\r\n`);
        return;
      }

      ws = new WebSocket(
        buildTerminalWsUrl({
          restApiUrl: environment.getAGPTServerApiUrl(),
          sessionId,
          token,
        }),
      );
      ws.binaryType = "arraybuffer";

      ws.onopen = () => {
        fitAddon?.fit();
        sendResize();
      };
      ws.onmessage = (event) => {
        if (typeof event.data === "string") term?.write(event.data);
        else term?.write(new Uint8Array(event.data));
      };
      ws.onclose = () => {
        if (!disposed) setIsClosed(true);
      };

      term.onData((data) => {
        if (ws?.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ type: "input", data }));
        }
      });

      resizeObserver = new ResizeObserver(() => {
        fitAddon?.fit();
        sendResize();
      });
      resizeObserver.observe(element);
    }

    setIsClosed(false);
    start();

    return () => {
      disposed = true;
      resizeObserver?.disconnect();
      ws?.close();
      term?.dispose();
    };
  }, [sessionId, reconnectKey]);

  function reconnect() {
    setReconnectKey((key) => key + 1);
  }

  return { containerRef, isClosed, reconnect };
}
