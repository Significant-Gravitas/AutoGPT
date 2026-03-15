/**
 * Partial copy of types/google.ts from @next/third-parties.
 * Original source file: https://github.com/vercel/next.js/blob/cb49b483f1e2f63dcf6880765585830b48154e29/packages/third-parties/src/types/google.ts
 */

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
