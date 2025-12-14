/**
 * EY Brand Design System
 *
 * Central export for all EY brand design tokens and specifications.
 * Import this file to access the complete design system.
 */

// Color tokens
export {
  eyColors,
  eySemanticColors,
  eyTailwindColors,
  type EYColorKey,
  type EYStatusColor,
} from './colors';

// Typography
export {
  eyFontFamily,
  eyFontSize,
  eyLineHeight,
  eyFontWeight,
  eyTypography,
  eyTailwindFonts,
  type EYTypographyKey,
} from './typography';

// Icons
export {
  eyIcons,
  eyIconNames,
  getEYIconPath,
  type EYIconName,
  type EYIconVariant,
} from './icons';

// Component specifications
export {
  headerSpec,
  leftNavSpec,
  navItemSpec,
  kpiTileSpec,
  tableSpec,
  buttonSpec,
  badgeSpec,
  tabSpec,
  modalSpec,
  progressSpec,
  chartSpec,
  cardSpec,
  inputSpec,
  dropdownSpec,
  tooltipSpec,
  toastSpec,
  type ComponentSpec,
} from './components';

// Layout constants
export const eyLayout = {
  containerWidth: 1200,
  headerHeight: 56,
  navWidth: 220,
  navCollapsedWidth: 72,
  sidePadding: 24,
  gridGutter: 24,
  baseline: 8,
  kpiTileHeight: 96,
} as const;

// Spacing scale (based on 8px baseline)
export const eySpacing = {
  0: 0,
  1: 4,
  2: 8,
  3: 12,
  4: 16,
  5: 20,
  6: 24,
  8: 32,
  10: 40,
  12: 48,
  16: 64,
} as const;

// Border radius tokens
export const eyRadius = {
  sm: 4,
  button: 6,
  default: 8,
  md: 12,
  lg: 16,
  full: 9999,
} as const;

// Shadow tokens
export const eyShadow = {
  sm: '0 1px 2px rgba(15, 15, 15, 0.04)',
  default: '0 2px 6px rgba(15, 15, 15, 0.06)',
  md: '0 4px 12px rgba(15, 15, 15, 0.08)',
  lg: '0 8px 24px rgba(15, 15, 15, 0.12)',
  xl: '0 16px 48px rgba(15, 15, 15, 0.16)',
} as const;

// Z-index layers
export const eyZIndex = {
  dropdown: 100,
  sticky: 200,
  fixed: 300,
  modalBackdrop: 400,
  modal: 500,
  tooltip: 600,
  toast: 700,
} as const;

// Transition presets
export const eyTransition = {
  fast: '150ms ease-in-out',
  default: '200ms ease-in-out',
  slow: '300ms ease-in-out',
} as const;

// Logo paths
export const eyLogos = {
  primary48: '/ey-brand/logos/ey-logo-primary-48.svg',
  primary120: '/ey-brand/logos/ey-logo-primary-120.svg',
  stacked48: '/ey-brand/logos/ey-logo-stacked-48.svg',
  stacked120: '/ey-brand/logos/ey-logo-stacked-120.svg',
  acaTrackMono: '/ey-brand/logos/aca-track-logo-mono.svg',
  acaTrackDark: '/ey-brand/logos/aca-track-logo-dark.svg',
} as const;
