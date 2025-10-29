declare global {
  interface Window {
    gapi?: {
      load: (
        name: "picker",
        options: { callback?: () => void; onerror?: (error: unknown) => void },
      ) => void;
    };
    google?: {
      accounts?: {
        oauth2?: unknown;
      };
      picker?: {
        ViewId: {
          DOCS: string;
          DOCUMENTS: string;
          SPREADSHEETS: string;
          PRESENTATIONS: string;
          DOCS_IMAGES: string;
          FOLDERS: string;
        };
        Response: {
          ACTION: string;
          DOCUMENTS: string;
        };
        Action: {
          PICKED: string;
        };
        Document: {
          ID: string;
          NAME: string;
          MIME_TYPE: string;
          URL: string;
          ICON_URL: string;
        };
      };
    };
  }
}

export {};
