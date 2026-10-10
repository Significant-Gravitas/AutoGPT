import { z } from "zod";

const response = z.object({
  id: z.string(),
  permission: z.enum(["granted", "denied", "unavailable", "disabled"]),
  provider: z.enum(["apns", "fcm"]).optional(),
  token: z.string().optional(),
  environment: z.enum(["sandbox", "production"]).optional(),
  binding_id: z.string().uuid().optional(),
});

export type NativePushResponse = z.infer<typeof response>;

declare global {
  interface Window {
    AutoGPTPush?: {
      postMessage: (message: string) => void;
      onmessage?: (event: MessageEvent<string>) => void;
    };
    webkit?: {
      messageHandlers?: {
        AutoGPTPush?: { postMessage: (message: string) => void };
      };
    };
  }
}

function bridge() {
  if (typeof window === "undefined") return undefined;
  return window.AutoGPTPush ?? window.webkit?.messageHandlers?.AutoGPTPush;
}

export function hasNativePush() {
  return !!bridge();
}

export function requestNativePush(
  action: "status" | "enable" | "disable",
  accountID: string,
): Promise<NativePushResponse> {
  const channel = bridge();
  if (!channel) return Promise.resolve({ id: "", permission: "unavailable" });
  if (window.AutoGPTPush)
    window.AutoGPTPush.onmessage = (event) => {
      try {
        window.dispatchEvent(
          new CustomEvent("autogpt-native-push", {
            detail: JSON.parse(event.data),
          }),
        );
      } catch {}
    };
  return new Promise((resolve, reject) => {
    const id = crypto.randomUUID();
    const timeout = window.setTimeout(
      () => finish(new Error("Notification setup timed out. Try again.")),
      20_000,
    );
    function finish(value: NativePushResponse | Error) {
      window.clearTimeout(timeout);
      window.removeEventListener("autogpt-native-push", receive);
      if (value instanceof Error) reject(value);
      else resolve(value);
    }
    function receive(event: Event) {
      if (!(event instanceof CustomEvent)) return;
      const parsed = response.safeParse(event.detail);
      if (parsed.success && parsed.data.id === id) finish(parsed.data);
    }
    window.addEventListener("autogpt-native-push", receive);
    try {
      channel.postMessage(
        JSON.stringify({ id, action, account_id: accountID }),
      );
    } catch {
      finish(
        new Error("Notification setup is unavailable. Reload and try again."),
      );
    }
  });
}
