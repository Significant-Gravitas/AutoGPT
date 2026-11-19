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
  sky: "#38BDF8",
  avatarBorder: "rgba(255,255,255,0.12)",
  avatarTintBase: "#17171B",
  // Laid over a page pushed aside by a drawer, in the page's own colour.
  pageWash: "rgba(11,11,13,0.55)",
  pageEdge: "rgba(255,255,255,0.10)",
  // The menu and the chat panel sit a step lighter than the page.
  menuBackground: "#141417",
  menuRowActive: "#26262D",
  menuButton: "#232329",
  panelBackground: "#141417",
  panelCard: "#1F1F25",
  panelTabTrack: "#26262D",
  // The chat's round header buttons and name pill, over the thread.
  floatingButton: "#1F1F25",
  floatingRing: "#33333C",
  // Lines between rows inside a panel card.
  panelDivider: "#2A2A30",
  // The marketplace's round hire button and pinned hire pill.
  actionStrong: "#F4F4F5",
  onActionStrong: "#141414",
  // The tick that marks an expert already on the team.
  hiredBadge: "#16301F",
  onHiredBadge: "#4ADE80",
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
  sky: "#0EA5E9",
  avatarBorder: "rgba(20,20,30,0.10)",
  avatarTintBase: "#FFFFFF",
  pageWash: "rgba(254,254,254,0.55)",
  pageEdge: "rgba(20,20,30,0.06)",
  menuBackground: "#F3F3F4",
  menuRowActive: "#E4E4E7",
  menuButton: "#FFFFFF",
  panelBackground: "#F3F3F4",
  panelCard: "#FFFFFF",
  panelTabTrack: "#E6E6E9",
  floatingButton: "#FFFFFF",
  floatingRing: "rgba(0,0,0,0.08)",
  panelDivider: "#EFEFF1",
  actionStrong: "#141414",
  onActionStrong: "#FFFFFF",
  hiredBadge: "#E7F6EC",
  onHiredBadge: "#16A34A",
};

export const palettes: Record<Scheme, Colors> = { dark, light };

export const avatarTintPercent: Record<Scheme, number> = { dark: 16, light: 24 };

export const spacing = { xs: 4, sm: 8, md: 12, lg: 16, xl: 24, xxl: 32 } as const;

// `page` matches the phone's own corners, for a page pushed aside by a drawer.
export const radius = { sm: 8, md: 12, lg: 18, page: 40, pill: 999 } as const;

// Light theme only: in dark, surfaces part by colour and shadows don't show.
export const shadows = {
  // The chat header's round buttons and name pill.
  floating: {
    shadowColor: "#000000",
    shadowOpacity: 0.1,
    shadowRadius: 5,
    shadowOffset: { width: 0, height: 2 },
    elevation: 3,
  },
  // The menu's search and avatar buttons.
  menuButton: {
    shadowColor: "#000000",
    shadowOpacity: 0.1,
    shadowRadius: 3,
    shadowOffset: { width: 0, height: 1 },
    elevation: 2,
  },
  // The chosen tab in the panel's tab track.
  selectedTab: {
    shadowColor: "#000000",
    shadowOpacity: 0.08,
    shadowRadius: 1.5,
    shadowOffset: { width: 0, height: 1 },
    elevation: 1,
  },
} as const;

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
