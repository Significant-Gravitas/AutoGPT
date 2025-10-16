/* eslint-disable @typescript-eslint/ban-ts-comment */
// Utility helpers for Google Drive Picker
import { loadScript } from "@/services/scripts/scripts";

export async function loadGoogleAPIPicker(): Promise<void> {
  validateWindow();

  await loadScript("https://apis.google.com/js/api.js");

  // @ts-ignore `window.gapi` is defined by the script above
  const googleAPI = window.gapi;
  if (!googleAPI) {
    throw new Error(
      "Google AIP not available after loading https://apis.google.com/js/api.js",
    );
  }

  await new Promise<void>((resolve, reject) => {
    try {
      googleAPI.load("picker", { callback: resolve });
    } catch (e) {
      reject(e);
    }
  });
}

export async function loadGoogleIdentityServices(): Promise<void> {
  if (typeof window === "undefined") {
    throw new Error("Google Identity Services cannot load on server");
  }

  await loadScript("https://accounts.google.com/gsi/client");

  // @ts-ignore `window.google` is defined by the script above
  const google = window.google;
  if (!google?.accounts?.oauth2) {
    throw new Error("Google Identity Services not available");
  }
}

export type GooglePickerView =
  | "DOCS"
  | "DOCUMENTS"
  | "SPREADSHEETS"
  | "PRESENTATIONS"
  | "DOCS_IMAGES"
  | "FOLDERS";

export function mapViewId(view: GooglePickerView): any {
  validateWindow();

  // @ts-ignore `window.google.picker` is defined by the script above
  const gp = window.google?.picker;
  if (!gp) {
    throw new Error("google.picker is not available");
  }

  switch (view) {
    case "DOCS":
      return gp.ViewId.DOCS;
    case "DOCUMENTS":
      return gp.ViewId.DOCUMENTS;
    case "SPREADSHEETS":
      return gp.ViewId.SPREADSHEETS;
    case "PRESENTATIONS":
      return gp.ViewId.PRESENTATIONS;
    case "DOCS_IMAGES":
      return gp.ViewId.DOCS_IMAGES;
    case "FOLDERS":
      return gp.ViewId.FOLDERS;
    default:
      return gp.ViewId.DOCS;
  }
}

export function scopesIncludeDrive(scopes: string[]): boolean {
  const set = new Set(scopes);
  if (set.has("https://www.googleapis.com/auth/drive")) return true;
  if (set.has("https://www.googleapis.com/auth/drive.readonly")) return true;
  return false;
}

export type NormalizedPickedFile = {
  id: string;
  name?: string;
  mimeType?: string;
  url?: string;
  iconUrl?: string;
};

export function normalizePickerResponse(data: any): NormalizedPickedFile[] {
  validateWindow();

  // @ts-ignore `window.google.picker` should have been loaded
  const gp = window.google?.picker;
  if (!gp) return [];
  if (!data || data[gp.Response.ACTION] !== gp.Action.PICKED) return [];
  const docs = data[gp.Response.DOCUMENTS] || [];
  return docs.map((doc: any) => ({
    id: doc[gp.Document.ID],
    name: doc[gp.Document.NAME],
    mimeType: doc[gp.Document.MIME_TYPE],
    url: doc[gp.Document.URL],
    iconUrl: doc[gp.Document.ICON_URL],
  }));
}

export type OAuthPopupResultMessage = { message_type: "oauth_popup_result" } & (
  | {
      success: true;
      code: string;
      state: string;
    }
  | {
      success: false;
      message: string;
    }
);

export function startOAuthPopupFlow(
  loginUrl: string,
  expectedState: string,
  timeoutMs: number = 5 * 60 * 1000,
): { promise: Promise<{ code: string; state: string }>; abort: () => void } {
  validateWindow();

  const popup = window.open(loginUrl, "_blank", "popup=true");
  if (!popup) {
    throw new Error(
      "Failed to open popup window. Please allow popups for this site.",
    );
  }

  const controller = new AbortController();

  const promise = new Promise<{ code: string; state: string }>(
    (resolve, reject) => {
      controller.signal.onabort = () => {
        try {
          popup.close();
        } catch {}
      };

      function handleMessage(e: MessageEvent<OAuthPopupResultMessage>) {
        const data = e.data;
        if (
          typeof data !== "object" ||
          !data ||
          !("message_type" in data) ||
          data.message_type !== "oauth_popup_result"
        )
          return;

        if (!data.success) {
          controller.abort();
          return reject(new Error(data.message));
        }

        if (data.state !== expectedState) {
          controller.abort();
          return reject(new Error("Invalid state token received"));
        }

        controller.abort();
        resolve({ code: data.code, state: data.state });
      }

      window.addEventListener("message", handleMessage, {
        signal: controller.signal,
      });

      setTimeout(() => {
        controller.abort();
        reject(new Error("OAuth flow timed out"));
      }, timeoutMs);
    },
  );

  return { promise, abort: () => controller.abort() };
}

function validateWindow() {
  if (typeof window === "undefined") {
    throw new Error("Google Picker cannot load on server");
  }
}
