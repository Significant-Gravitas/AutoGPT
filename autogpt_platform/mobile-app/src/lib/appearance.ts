import * as SecureStore from "expo-secure-store";
import { Appearance, Platform } from "react-native";

export type AppearanceChoice = "system" | "light" | "dark";

const STORAGE_KEY = "autogpt.appearance";

export function isAppearanceChoice(value: unknown): value is AppearanceChoice {
  return value === "system" || value === "light" || value === "dark";
}

export function readAppearance(): AppearanceChoice {
  try {
    const stored = SecureStore.getItem(STORAGE_KEY);
    return isAppearanceChoice(stored) ? stored : "system";
  } catch {
    return "system";
  }
}

/** Overrides the scheme `useColorScheme` reports, app-wide; "system" follows the phone. */
export function applyAppearance(choice: AppearanceChoice) {
  // react-native-web has no setColorScheme; the browser's own setting holds there.
  if (Platform.OS === "web") return;
  Appearance.setColorScheme(choice === "system" ? "unspecified" : choice);
}

export function saveAppearance(choice: AppearanceChoice) {
  applyAppearance(choice);
  try {
    SecureStore.setItem(STORAGE_KEY, choice);
  } catch {
    // The choice still holds until the app restarts.
  }
}
