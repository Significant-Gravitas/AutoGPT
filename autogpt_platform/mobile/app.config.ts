import type { ExpoConfig } from "expo/config";

const config: ExpoConfig = {
  name: "AutoGPT",
  slug: "autogpt-native-experiment",
  version: "0.1.0",
  scheme: "autogpt-experiment",
  orientation: "default",
  userInterfaceStyle: "automatic",
  icon: "./assets/icon.png",
  ios: {
    bundleIdentifier: "com.agpt.mobile.experimental",
    supportsTablet: false,
    infoPlist: {
      ITSAppUsesNonExemptEncryption: false,
      NSLocalNetworkUsageDescription:
        "Connect to your own AutoGPT server on your local network.",
      NSAppTransportSecurity: { NSAllowsLocalNetworking: true },
    },
  },
  android: { package: "com.agpt.mobile.experimental" },
  plugins: [
    "expo-router",
    [
      "expo-build-properties",
      {
        ios: {
          deploymentTarget: "17.0",
          usePrecompiledModules: true,
          enableSceneSupport: true,
        },
      },
    ],
    [
      "expo-splash-screen",
      { backgroundColor: "#ffffff", dark: { backgroundColor: "#111111" } },
    ],
    "expo-secure-store",
    "expo-web-browser",
    "expo-notifications",
    "expo-document-picker",
    "expo-image",
    "expo-sharing",
    [
      "expo-video",
      { supportsBackgroundPlayback: false, supportsPictureInPicture: false },
    ],
    [
      "expo-audio",
      {
        microphonePermission: false,
        recordAudioAndroid: false,
        enableBackgroundPlayback: false,
        enableBackgroundRecording: false,
      },
    ],
    "expo-sqlite",
    "expo-iap",
  ],
  updates: { enabled: false },
};

export default config;
