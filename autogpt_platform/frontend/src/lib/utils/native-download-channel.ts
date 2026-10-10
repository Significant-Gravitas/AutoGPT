import {
  NativeDownloadBridge,
  readNativeDownloadReply,
} from "./native-download-protocol";

interface PendingReply {
  type: string;
  index?: number;
  resolve: () => void;
  reject: (error: Error) => void;
  timer: ReturnType<typeof setTimeout>;
}

export class NativeDownloadChannel {
  private pending?: PendingReply;
  private failure?: Error;
  private started = false;
  private remoteClosed = false;
  private cancellationSent = false;
  private previousHandler: NativeDownloadBridge["onmessage"];
  private readonly onMessage = this.receive.bind(this);
  private readonly onAbort = this.abort.bind(this);

  constructor(
    private bridge: NativeDownloadBridge,
    private id: string,
    private signal?: AbortSignal,
  ) {
    this.previousHandler = bridge.onmessage;
    bridge.onmessage = this.onMessage;
    window.addEventListener("pagehide", this.onAbort);
    signal?.addEventListener("abort", this.onAbort, { once: true });
    if (signal?.aborted) this.abort();
  }

  request(message: Record<string, unknown>, type: string, timeout: number) {
    if (this.failure) return Promise.reject(this.failure);
    if (this.pending)
      return Promise.reject(new Error("A download reply is still pending."));
    return new Promise<void>((resolve, reject) => {
      this.pending = {
        type,
        index:
          type === "ack" && typeof message.index === "number"
            ? message.index
            : undefined,
        resolve,
        reject,
        timer: setTimeout(() => {
          this.fail(new Error("Download timed out. Please try again."));
          this.cancel();
        }, timeout),
      };
      try {
        if (message.type === "start") this.started = true;
        this.bridge.postMessage(JSON.stringify({ ...message, id: this.id }));
      } catch {
        this.fail(new Error("Could not start the native download."));
      }
    });
  }

  cancel() {
    if (!this.started || this.remoteClosed || this.cancellationSent) return;
    this.cancellationSent = true;
    try {
      this.bridge.postMessage(JSON.stringify({ type: "cancel", id: this.id }));
    } catch {}
  }

  close() {
    window.removeEventListener("pagehide", this.onAbort);
    this.signal?.removeEventListener("abort", this.onAbort);
    if (this.bridge.onmessage === this.onMessage) {
      this.bridge.onmessage = this.previousHandler;
    }
    if (this.pending) clearTimeout(this.pending.timer);
    this.pending = undefined;
  }

  private abort() {
    this.fail(new DOMException("Download cancelled.", "AbortError"));
    this.cancel();
  }

  private fail(error: Error) {
    this.failure ??= error;
    const pending = this.pending;
    this.pending = undefined;
    if (!pending) return;
    clearTimeout(pending.timer);
    pending.reject(this.failure);
  }

  private receive(event: { data: unknown }) {
    const reply = readNativeDownloadReply(event.data);
    if (!reply || reply.id !== this.id) {
      this.previousHandler?.call(this.bridge, event);
      return;
    }
    if (reply.type === "error" || reply.type === "cancelled") {
      this.remoteClosed = true;
      this.fail(
        reply.type === "cancelled"
          ? new DOMException("Download cancelled.", "AbortError")
          : new Error(reply.message || "Could not save the file."),
      );
      return;
    }
    const pending = this.pending;
    if (!pending || reply.type !== pending.type) return;
    if (pending.type === "ack" && reply.index !== pending.index) return;
    if (reply.type === "complete") this.remoteClosed = true;
    clearTimeout(pending.timer);
    this.pending = undefined;
    pending.resolve();
  }
}
