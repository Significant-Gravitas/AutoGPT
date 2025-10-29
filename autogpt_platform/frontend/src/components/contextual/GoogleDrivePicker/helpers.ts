import { BlockIOCredentialsSubSchema } from "@/lib/autogpt-server-api";
import { loadScript } from "@/services/scripts/scripts";

export async function loadGoogleAPIPicker(): Promise<void> {
  validateWindow();

  await loadScript("https://apis.google.com/js/api.js");

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

function validateWindow() {
  if (typeof window === "undefined") {
    throw new Error("Google Picker cannot load on server");
  }
}

export function getCredentialsSchema(scopes: string[]) {
  return {
    type: "object",
    title: "Google Drive",
    description: "Google OAuth needed to access Google Drive",
    properties: {},
    required: [],
    credentials_provider: ["google"],
    credentials_types: ["oauth2"],
    credentials_scopes: scopes,
    secret: true,
  } satisfies BlockIOCredentialsSubSchema;
}
