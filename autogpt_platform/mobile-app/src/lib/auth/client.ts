import { expoClient } from "@better-auth/expo/client";
import { createAuthClient } from "better-auth/react";
import * as SecureStore from "expo-secure-store";

import { APP_SCHEME, FRONTEND_URL } from "@/lib/env";

export const authClient = createAuthClient({
  baseURL: FRONTEND_URL,
  plugins: [
    expoClient({
      scheme: APP_SCHEME,
      storagePrefix: "autogpt",
      storage: SecureStore,
    }),
  ],
});
