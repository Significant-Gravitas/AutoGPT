import { vi } from "vitest";

export interface NativeTestMessage {
  type: string;
  id: string;
  filename?: string;
  mimeType?: string;
  size?: number;
  index?: number;
  data?: string;
  message?: string;
}

export function installDownloadBridge() {
  const messages: NativeTestMessage[] = [];
  const previousHandler = vi.fn();
  const bridge = {
    onmessage: previousHandler as ((event: { data: unknown }) => void) | null,
    postMessage: vi.fn((data: string) => {
      messages.push(JSON.parse(data));
    }),
  };
  Object.defineProperty(window, "AutoGPTDownloads", {
    configurable: true,
    value: bridge,
  });

  function reply(message: NativeTestMessage) {
    bridge.onmessage?.({ data: JSON.stringify(message) });
  }

  return { bridge, messages, previousHandler, reply };
}
