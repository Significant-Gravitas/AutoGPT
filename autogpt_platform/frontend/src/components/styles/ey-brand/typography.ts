/**
 * EY Brand Typography Configuration
 *
 * Preferred typeface: EY Hellix
 * Fallback: Inter or Helvetica Neue
 */

export const eyFontFamily = {
  primary: '"EY Hellix", Inter, "Helvetica Neue", Arial, sans-serif',
  mono: '"JetBrains Mono", "Fira Code", Consolas, monospace',
} as const;

export const eyFontSize = {
  h1: '28px',
  h2: '20px',
  h3: '16px',
  body: '14px',
  small: '12px',
  xs: '10px',
} as const;

export const eyLineHeight = {
  h1: '36px',
  h2: '28px',
  h3: '22px',
  body: '20px',
  small: '18px',
  xs: '14px',
} as const;

export const eyFontWeight = {
  regular: 400,
  medium: 500,
  semibold: 600,
  bold: 700,
} as const;

// Typography presets for common use cases
export const eyTypography = {
  h1: {
    fontFamily: eyFontFamily.primary,
    fontSize: eyFontSize.h1,
    lineHeight: eyLineHeight.h1,
    fontWeight: eyFontWeight.bold,
  },
  h2: {
    fontFamily: eyFontFamily.primary,
    fontSize: eyFontSize.h2,
    lineHeight: eyLineHeight.h2,
    fontWeight: eyFontWeight.semibold,
  },
  h3: {
    fontFamily: eyFontFamily.primary,
    fontSize: eyFontSize.h3,
    lineHeight: eyLineHeight.h3,
    fontWeight: eyFontWeight.semibold,
  },
  body: {
    fontFamily: eyFontFamily.primary,
    fontSize: eyFontSize.body,
    lineHeight: eyLineHeight.body,
    fontWeight: eyFontWeight.regular,
  },
  bodyMedium: {
    fontFamily: eyFontFamily.primary,
    fontSize: eyFontSize.body,
    lineHeight: eyLineHeight.body,
    fontWeight: eyFontWeight.medium,
  },
  small: {
    fontFamily: eyFontFamily.primary,
    fontSize: eyFontSize.small,
    lineHeight: eyLineHeight.small,
    fontWeight: eyFontWeight.regular,
  },
  mono: {
    fontFamily: eyFontFamily.mono,
    fontSize: eyFontSize.small,
    lineHeight: eyLineHeight.small,
    fontWeight: eyFontWeight.regular,
  },
  tableSmall: {
    fontFamily: eyFontFamily.mono,
    fontSize: eyFontSize.small,
    lineHeight: eyLineHeight.small,
    fontWeight: eyFontWeight.regular,
  },
  label: {
    fontFamily: eyFontFamily.primary,
    fontSize: eyFontSize.small,
    lineHeight: eyLineHeight.small,
    fontWeight: eyFontWeight.medium,
    textTransform: 'uppercase' as const,
    letterSpacing: '0.5px',
  },
  kpiValue: {
    fontFamily: eyFontFamily.primary,
    fontSize: '32px',
    lineHeight: '40px',
    fontWeight: eyFontWeight.bold,
  },
  kpiLabel: {
    fontFamily: eyFontFamily.primary,
    fontSize: eyFontSize.small,
    lineHeight: eyLineHeight.small,
    fontWeight: eyFontWeight.medium,
  },
} as const;

// Tailwind-compatible font configuration
export const eyTailwindFonts = {
  fontFamily: {
    'ey-sans': ['"EY Hellix"', 'Inter', '"Helvetica Neue"', 'Arial', 'sans-serif'],
    'ey-mono': ['"JetBrains Mono"', '"Fira Code"', 'Consolas', 'monospace'],
  },
  fontSize: {
    'ey-h1': ['28px', { lineHeight: '36px', fontWeight: '700' }],
    'ey-h2': ['20px', { lineHeight: '28px', fontWeight: '600' }],
    'ey-h3': ['16px', { lineHeight: '22px', fontWeight: '600' }],
    'ey-body': ['14px', { lineHeight: '20px', fontWeight: '400' }],
    'ey-small': ['12px', { lineHeight: '18px', fontWeight: '400' }],
    'ey-kpi': ['32px', { lineHeight: '40px', fontWeight: '700' }],
  },
};

export type EYTypographyKey = keyof typeof eyTypography;
