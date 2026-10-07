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
        oauth2?: {
          initTokenClient: (options: {
            client_id: string;
            scope: string;
            callback: (response: any) => void;
          }) => {
            // Minimal surface we use
            requestAccessToken: (opts: { prompt: string }) => void;
            callback?: (response: any) => void;
          };
        };
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
        Feature: {
          NAV_HIDDEN: string;
          MULTISELECT_ENABLED: string;
        };
        DocsViewMode: {
          LIST: string;
        };
        PickerBuilder: new () => {
          setOAuthToken: (token: string) => any;
          setDeveloperKey: (key: string) => any;
          setAppId: (id: string) => any;
          setCallback: (cb: (data: any) => void) => any;
          enableFeature: (feature: string) => any;
          addView: (view: any) => any;
          build: () => { setVisible: (visible: boolean) => void };
        };
        DocsView: new (viewId: any) => { setMode: (mode: string) => any };
      };
    };
  }
}

export {};
