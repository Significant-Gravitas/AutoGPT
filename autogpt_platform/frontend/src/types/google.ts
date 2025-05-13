type JSONValue =
  | string
  | number
  | boolean
  | JSONValue[]
  | {
      [key: string]: JSONValue;
    };

export type GTMParams = {
  gtmId: string;
  gtmScriptUrl?: string;
  dataLayer?: {
    [key: string]: JSONValue;
  };
  dataLayerName?: string;
  auth?: string;
  preview?: string;
  nonce?: string;
};

export type GAParams = {
  gaId: string;
  dataLayerName?: string;
  debugMode?: boolean;
  nonce?: string;
};
