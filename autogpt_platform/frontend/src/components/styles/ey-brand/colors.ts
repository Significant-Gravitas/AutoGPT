/**
 * EY Brand Color Tokens
 *
 * NOTE: These are prototype-ready values. Confirm exact values with EY Brand Guidelines.
 * Replace #FFD100 with official EY Yellow Pantone/HEX value.
 */

export const eyColors = {
  // Primary Colors
  primary: {
    yellow: '#FFD100', // EY Yellow (replace with official)
    black: '#000000',  // EY Black
  },

  // Neutrals
  neutral: {
    darkGray: '#2B2B2B',
    mediumGray: '#6B6B6B',
    lightGray: '#F4F6F8', // Background
    white: '#FFFFFF',
  },

  // Status Colors
  status: {
    success: '#12A454',    // Verified / Clean
    warning: '#FFB300',    // At Risk
    critical: '#D62828',   // Error / Critical
    info: '#2F80ED',       // Info / Neutral
  },

  // Extended Palette (for charts, pods, etc.)
  extended: {
    podA: '#2F80ED',       // Blue
    podB: '#9B51E0',       // Purple
    podC: '#27AE60',       // Green
    podD: '#F2994A',       // Orange
    accent1: '#56CCF2',    // Light Blue
    accent2: '#BB6BD9',    // Light Purple
  },
} as const;

// Semantic color mapping for component variants
export const eySemanticColors = {
  // Text colors
  text: {
    primary: eyColors.neutral.darkGray,
    secondary: eyColors.neutral.mediumGray,
    inverse: eyColors.neutral.white,
    brand: eyColors.primary.black,
  },

  // Background colors
  background: {
    primary: eyColors.neutral.white,
    secondary: eyColors.neutral.lightGray,
    brand: eyColors.primary.yellow,
    dark: eyColors.primary.black,
  },

  // Border colors
  border: {
    default: '#E5E7EB',
    hover: eyColors.neutral.mediumGray,
    focus: eyColors.primary.yellow,
    error: eyColors.status.critical,
  },

  // Interactive states
  interactive: {
    primary: eyColors.primary.yellow,
    primaryHover: '#E6BC00', // Darker yellow
    secondary: eyColors.neutral.darkGray,
    secondaryHover: eyColors.primary.black,
    danger: eyColors.status.critical,
    dangerHover: '#B82222',
  },

  // Status backgrounds (lighter versions for badges/alerts)
  statusBackground: {
    success: '#E8F5E9',
    warning: '#FFF8E1',
    critical: '#FFEBEE',
    info: '#E3F2FD',
  },

  // Row states for tables
  rowState: {
    normal: 'transparent',
    warning: eyColors.status.warning,
    critical: eyColors.status.critical,
    verified: eyColors.status.success,
  },
} as const;

// Tailwind-compatible color configuration
export const eyTailwindColors = {
  'ey-yellow': {
    DEFAULT: '#FFD100',
    50: '#FFF9E6',
    100: '#FFF3CC',
    200: '#FFE799',
    300: '#FFDB66',
    400: '#FFCF33',
    500: '#FFD100',
    600: '#E6BC00',
    700: '#B38F00',
    800: '#806600',
    900: '#4D3D00',
  },
  'ey-black': '#000000',
  'ey-gray': {
    DEFAULT: '#6B6B6B',
    50: '#F9FAFB',
    100: '#F4F6F8',
    200: '#E5E7EB',
    300: '#D1D5DB',
    400: '#9CA3AF',
    500: '#6B6B6B',
    600: '#4B5563',
    700: '#374151',
    800: '#2B2B2B',
    900: '#1F1F1F',
  },
  'ey-success': '#12A454',
  'ey-warning': '#FFB300',
  'ey-critical': '#D62828',
  'ey-info': '#2F80ED',
};

export type EYColorKey = keyof typeof eyColors;
export type EYStatusColor = keyof typeof eyColors.status;
