import type { ConfigContext, ExpoConfig } from "expo/config";

// app.json holds the store app. Every other build is "AutoGPT Dev", with its own
// ID and link scheme, so both apps fit on one phone and a sign-in started in one
// never comes back to the other. The web picks the matching scheme in
// frontend/src/app/(platform)/auth/integrations/native-return.ts.
export default function appConfig({ config }: ConfigContext): ExpoConfig {
  const base = config as ExpoConfig;
  if (process.env.APP_VARIANT === "production") return base;

  return {
    ...base,
    name: `${base.name} Dev`,
    scheme: `${base.scheme}-dev`,
    ios: {
      ...base.ios,
      bundleIdentifier: `${base.ios?.bundleIdentifier}.dev`,
    },
    android: {
      ...base.android,
      package: `${base.android?.package}.dev`,
    },
  };
}
