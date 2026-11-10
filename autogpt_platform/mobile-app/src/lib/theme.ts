export const colors = {
  background: "#0B0B0D",
  surface: "#16161A",
  surfaceRaised: "#1F1F25",
  surfacePressed: "#2A2A32",
  border: "#2A2A32",
  text: "#F4F4F5",
  textSecondary: "#A1A1AA",
  textTertiary: "#71717A",
  brand: "#7733F5",
  brandSoft: "#2A1A52",
  onBrand: "#FFFFFF",
  error: "#F87171",
  errorSurface: "#3A1616",
  success: "#4ADE80",
} as const;

export const spacing = { xs: 4, sm: 8, md: 12, lg: 16, xl: 24, xxl: 32 } as const;

export const radius = { sm: 8, md: 12, lg: 18, pill: 999 } as const;

export const type = {
  title: { fontSize: 28, fontWeight: "600", lineHeight: 34 },
  heading: { fontSize: 18, fontWeight: "600", lineHeight: 24 },
  body: { fontSize: 16, fontWeight: "400", lineHeight: 23 },
  label: { fontSize: 15, fontWeight: "500", lineHeight: 20 },
  small: { fontSize: 13, fontWeight: "400", lineHeight: 18 },
} as const;
