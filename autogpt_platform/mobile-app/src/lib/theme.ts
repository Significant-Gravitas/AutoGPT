import { StyleSheet, useColorScheme } from "react-native";

export type Scheme = "dark" | "light";

const dark = {
  background: "#0B0B0D",
  surface: "#16161A",
  surfaceRaised: "#1F1F25",
  surfacePressed: "#2A2A32",
  border: "#2A2A32",
  borderBase: "#33333C",
  borderStrong: "#44444F",
  text: "#F4F4F5",
  textSecondary: "#A1A1AA",
  textTertiary: "#71717A",
  brand: "#7733F5",
  brandSoft: "#2A1A52",
  accentText: "#C4A8FF",
  onBrand: "#FFFFFF",
  action: "#F4F4F5",
  onAction: "#141414",
  actionDisabled: "#2A2A32",
  onActionDisabled: "#71717A",
  error: "#F87171",
  errorSurface: "#3A1616",
  success: "#4ADE80",
  avatarBorder: "rgba(255,255,255,0.12)",
  avatarTintBase: "#17171B",
};

export type Colors = typeof dark;

// The web app's role tokens (frontend/src/app/tokens/theme.css).
const light: Colors = {
  background: "#FEFEFE",
  surface: "#F9F9FA",
  surfaceRaised: "#EFEFF0",
  surfacePressed: "#DADADC",
  border: "#E2E2E6",
  borderBase: "#DADADC",
  borderStrong: "#C5C5C9",
  text: "#141414",
  textSecondary: "#68686F",
  textTertiary: "#83838C",
  brand: "#7733F5",
  brandSoft: "#F5F3FF",
  accentText: "#6D28D9",
  onBrand: "#FFFFFF",
  action: "#3E3E43",
  onAction: "#FEFEFE",
  actionDisabled: "#DADADC",
  onActionDisabled: "#FEFEFE",
  error: "#DC2626",
  errorSurface: "#FEF5F5",
  success: "#16A34A",
  avatarBorder: "rgba(20,20,30,0.10)",
  avatarTintBase: "#FFFFFF",
};

export const palettes: Record<Scheme, Colors> = { dark, light };

export const avatarTintPercent: Record<Scheme, number> = { dark: 16, light: 24 };

export const spacing = { xs: 4, sm: 8, md: 12, lg: 16, xl: 24, xxl: 32 } as const;

export const radius = { sm: 8, md: 12, lg: 18, pill: 999 } as const;

export const type = {
  title: { fontSize: 28, fontWeight: "600", lineHeight: 34 },
  heading: { fontSize: 18, fontWeight: "600", lineHeight: 24 },
  body: { fontSize: 16, fontWeight: "400", lineHeight: 23 },
  label: { fontSize: 15, fontWeight: "500", lineHeight: 20 },
  small: { fontSize: 13, fontWeight: "400", lineHeight: 18 },
} as const;

export function useScheme(): Scheme {
  return useColorScheme() === "light" ? "light" : "dark";
}

export function useColors(): Colors {
  return palettes[useScheme()];
}

type StyleInput = Parameters<typeof StyleSheet.create>[0];

/** A `useStyles` hook whose sheet is built once per colour scheme. */
export function createStyles<S extends StyleInput>(
  build: (colors: Colors) => S,
): () => Readonly<S> {
  const sheets: Partial<Record<Scheme, Readonly<S>>> = {};
  function sheetFor(scheme: Scheme) {
    const sheet = sheets[scheme] ?? StyleSheet.create(build(palettes[scheme]));
    sheets[scheme] = sheet;
    return sheet;
  }
  return function useStyles() {
    return sheetFor(useScheme());
  };
}
