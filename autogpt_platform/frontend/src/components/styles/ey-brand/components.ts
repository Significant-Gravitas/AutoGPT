/**
 * EY Brand Component Specifications
 *
 * Design specifications for Figma component library with variants.
 * Each component should be built using Auto Layout and named layers for developer handoff.
 */

import { eyColors, eySemanticColors } from './colors';

// ============================================
// HEADER COMPONENT
// ============================================
export const headerSpec = {
  height: 56,
  background: eyColors.neutral.white,
  padding: { horizontal: 24 },
  shadow: '0 1px 3px rgba(15, 15, 15, 0.06)',
  zIndex: 300,
  elements: {
    logo: {
      position: 'left',
      size: 48,
      src: '/ey-brand/logos/aca-track-logo-dark.svg',
    },
    search: {
      position: 'center',
      width: 320,
      height: 36,
      placeholder: 'Search name / EIN',
      borderRadius: 6,
    },
    actions: {
      position: 'right',
      gap: 12,
      items: ['notifications', 'help', 'userDropdown'],
    },
  },
};

// ============================================
// LEFT NAV COMPONENT
// ============================================
export const leftNavSpec = {
  width: {
    expanded: 220,
    collapsed: 72,
  },
  background: eyColors.neutral.white,
  borderRight: `1px solid ${eySemanticColors.border.default}`,
  padding: { vertical: 16 },
  items: [
    { icon: 'dashboard', label: 'Overview', path: '/overview' },
    { icon: 'customers', label: 'Customers', path: '/customers' },
    { icon: 'onboarding', label: 'Onboarding & Migration', path: '/onboarding' },
    { icon: 'files', label: 'Files & Validation', path: '/files' },
    { icon: 'filings', label: 'Filings', path: '/filings' },
    { icon: 'print', label: 'Printing & Mailing', path: '/printing' },
    { icon: 'support', label: 'Support & SLA', path: '/support' },
    { icon: 'billing', label: 'Billing & Pricing', path: '/billing' },
    { icon: 'admin', label: 'Admin', path: '/admin' },
  ],
};

export const navItemSpec = {
  height: 44,
  padding: { horizontal: 16 },
  gap: 12,
  borderRadius: 8,
  states: {
    default: {
      background: 'transparent',
      color: eyColors.neutral.mediumGray,
    },
    hover: {
      background: eyColors.neutral.lightGray,
      color: eyColors.neutral.darkGray,
    },
    active: {
      background: eyColors.primary.yellow + '20', // 20% opacity
      color: eyColors.primary.black,
      fontWeight: 600,
      borderLeft: `3px solid ${eyColors.primary.yellow}`,
    },
  },
};

// ============================================
// KPI TILE COMPONENT
// ============================================
export const kpiTileSpec = {
  height: 96,
  minWidth: 180,
  padding: 16,
  borderRadius: 8,
  background: eyColors.neutral.white,
  shadow: '0 2px 6px rgba(15, 15, 15, 0.06)',
  states: {
    default: {
      border: '1px solid transparent',
    },
    hover: {
      border: `1px solid ${eyColors.primary.yellow}`,
      shadow: '0 4px 12px rgba(15, 15, 15, 0.1)',
      cursor: 'pointer',
    },
    active: {
      background: eyColors.primary.yellow + '10',
      border: `1px solid ${eyColors.primary.yellow}`,
    },
  },
  elements: {
    value: {
      fontSize: 32,
      fontWeight: 700,
      lineHeight: '40px',
      color: eyColors.neutral.darkGray,
    },
    label: {
      fontSize: 12,
      fontWeight: 500,
      lineHeight: '18px',
      color: eyColors.neutral.mediumGray,
    },
    sparkline: {
      height: 24,
      width: 64,
      strokeWidth: 2,
    },
    badge: {
      fontSize: 10,
      padding: '2px 6px',
      borderRadius: 4,
    },
  },
};

// ============================================
// TABLE COMPONENT
// ============================================
export const tableSpec = {
  borderRadius: 8,
  overflow: 'hidden',
  border: `1px solid ${eySemanticColors.border.default}`,
  header: {
    height: 48,
    background: eyColors.neutral.lightGray,
    fontSize: 12,
    fontWeight: 600,
    color: eyColors.neutral.mediumGray,
    textTransform: 'uppercase',
    letterSpacing: '0.5px',
  },
  row: {
    height: 56,
    padding: { horizontal: 16 },
    borderBottom: `1px solid ${eySemanticColors.border.default}`,
    states: {
      default: {
        background: eyColors.neutral.white,
      },
      hover: {
        background: eyColors.neutral.lightGray,
      },
      selected: {
        background: eyColors.primary.yellow + '10',
      },
      warning: {
        borderLeft: `4px solid ${eyColors.status.warning}`,
      },
      critical: {
        borderLeft: `4px solid ${eyColors.status.critical}`,
      },
      verified: {
        borderLeft: `4px solid ${eyColors.status.success}`,
      },
    },
  },
  pagination: {
    height: 52,
    background: eyColors.neutral.lightGray,
    fontSize: 14,
    rowsPerPage: 50,
  },
};

// ============================================
// BUTTON VARIANTS
// ============================================
export const buttonSpec = {
  height: {
    small: 32,
    medium: 36,
    large: 44,
  },
  padding: {
    small: { horizontal: 12 },
    medium: { horizontal: 16 },
    large: { horizontal: 20 },
  },
  borderRadius: 6,
  fontSize: 14,
  fontWeight: 500,
  gap: 8,
  variants: {
    primary: {
      background: eyColors.primary.yellow,
      color: eyColors.primary.black,
      border: 'none',
      hover: {
        background: '#E6BC00',
      },
      active: {
        background: '#CCB000',
      },
      disabled: {
        background: '#E5E7EB',
        color: '#9CA3AF',
      },
    },
    secondary: {
      background: 'transparent',
      color: eyColors.neutral.darkGray,
      border: `1px solid ${eySemanticColors.border.default}`,
      hover: {
        background: eyColors.neutral.lightGray,
        borderColor: eyColors.neutral.mediumGray,
      },
    },
    danger: {
      background: eyColors.status.critical,
      color: eyColors.neutral.white,
      border: 'none',
      hover: {
        background: '#B82222',
      },
    },
    ghost: {
      background: 'transparent',
      color: eyColors.neutral.darkGray,
      border: 'none',
      hover: {
        background: eyColors.neutral.lightGray,
      },
    },
  },
};

// ============================================
// BADGE / PILL VARIANTS
// ============================================
export const badgeSpec = {
  height: 24,
  padding: { horizontal: 8 },
  borderRadius: 4,
  fontSize: 12,
  fontWeight: 500,
  variants: {
    success: {
      background: eySemanticColors.statusBackground.success,
      color: eyColors.status.success,
    },
    warning: {
      background: eySemanticColors.statusBackground.warning,
      color: '#B87800', // Darker warning for contrast
    },
    critical: {
      background: eySemanticColors.statusBackground.critical,
      color: eyColors.status.critical,
    },
    info: {
      background: eySemanticColors.statusBackground.info,
      color: eyColors.status.info,
    },
    neutral: {
      background: eyColors.neutral.lightGray,
      color: eyColors.neutral.darkGray,
    },
    // Pod pills
    podA: {
      background: '#E3F2FD',
      color: '#2F80ED',
    },
    podB: {
      background: '#F3E5F5',
      color: '#9B51E0',
    },
    podC: {
      background: '#E8F5E9',
      color: '#27AE60',
    },
    podD: {
      background: '#FFF3E0',
      color: '#F2994A',
    },
  },
};

// ============================================
// TAB COMPONENT
// ============================================
export const tabSpec = {
  height: 48,
  gap: 24,
  borderBottom: `2px solid ${eySemanticColors.border.default}`,
  item: {
    padding: { horizontal: 4, vertical: 12 },
    fontSize: 14,
    fontWeight: 500,
    states: {
      inactive: {
        color: eyColors.neutral.mediumGray,
        borderBottom: '2px solid transparent',
      },
      active: {
        color: eyColors.primary.black,
        borderBottom: `2px solid ${eyColors.primary.yellow}`,
      },
      hover: {
        color: eyColors.neutral.darkGray,
      },
    },
  },
};

// ============================================
// MODAL / OVERLAY COMPONENT
// ============================================
export const modalSpec = {
  maxWidth: {
    small: 400,
    medium: 600,
    large: 800,
    fullWidth: 1000,
  },
  borderRadius: 12,
  background: eyColors.neutral.white,
  shadow: '0 16px 48px rgba(15, 15, 15, 0.2)',
  backdrop: 'rgba(0, 0, 0, 0.5)',
  header: {
    height: 64,
    padding: { horizontal: 24 },
    borderBottom: `1px solid ${eySemanticColors.border.default}`,
    fontSize: 18,
    fontWeight: 600,
  },
  content: {
    padding: 24,
    maxHeight: '60vh',
    overflow: 'auto',
  },
  footer: {
    height: 72,
    padding: { horizontal: 24 },
    borderTop: `1px solid ${eySemanticColors.border.default}`,
    gap: 12,
  },
};

// ============================================
// PROGRESS COMPONENTS
// ============================================
export const progressSpec = {
  circular: {
    sizes: {
      small: { diameter: 48, strokeWidth: 4 },
      medium: { diameter: 80, strokeWidth: 6 },
      large: { diameter: 120, strokeWidth: 8 },
    },
    trackColor: eyColors.neutral.lightGray,
    getProgressColor: (value: number) => {
      if (value >= 90) return eyColors.status.success;
      if (value >= 70) return eyColors.status.info;
      if (value >= 50) return eyColors.status.warning;
      return eyColors.status.critical;
    },
  },
  linear: {
    height: 8,
    borderRadius: 4,
    trackColor: eyColors.neutral.lightGray,
  },
};

// ============================================
// CHART SPECIFICATIONS
// ============================================
export const chartSpec = {
  colors: {
    primary: eyColors.primary.yellow,
    secondary: eyColors.status.info,
    success: eyColors.status.success,
    warning: eyColors.status.warning,
    critical: eyColors.status.critical,
    series: [
      '#2F80ED', // Blue
      '#9B51E0', // Purple
      '#27AE60', // Green
      '#F2994A', // Orange
      '#56CCF2', // Light Blue
      '#BB6BD9', // Light Purple
    ],
  },
  gridLines: {
    color: '#E5E7EB',
    dashArray: '4 4',
  },
  axis: {
    labelColor: eyColors.neutral.mediumGray,
    fontSize: 11,
  },
  tooltip: {
    background: eyColors.neutral.darkGray,
    color: eyColors.neutral.white,
    borderRadius: 4,
    padding: 8,
    fontSize: 12,
  },
};

// ============================================
// CARD COMPONENT
// ============================================
export const cardSpec = {
  padding: 24,
  borderRadius: 8,
  background: eyColors.neutral.white,
  shadow: '0 2px 6px rgba(15, 15, 15, 0.06)',
  border: 'none',
  variants: {
    elevated: {
      shadow: '0 4px 12px rgba(15, 15, 15, 0.08)',
    },
    outlined: {
      shadow: 'none',
      border: `1px solid ${eySemanticColors.border.default}`,
    },
    interactive: {
      cursor: 'pointer',
      hover: {
        shadow: '0 4px 12px rgba(15, 15, 15, 0.1)',
        borderColor: eyColors.primary.yellow,
      },
    },
  },
};

// ============================================
// INPUT COMPONENT
// ============================================
export const inputSpec = {
  height: 40,
  padding: { horizontal: 12 },
  borderRadius: 6,
  fontSize: 14,
  border: `1px solid ${eySemanticColors.border.default}`,
  background: eyColors.neutral.white,
  states: {
    default: {},
    focus: {
      border: `1px solid ${eyColors.primary.yellow}`,
      outline: `2px solid ${eyColors.primary.yellow}40`,
    },
    error: {
      border: `1px solid ${eyColors.status.critical}`,
    },
    disabled: {
      background: eyColors.neutral.lightGray,
      color: eyColors.neutral.mediumGray,
    },
  },
  placeholder: {
    color: eyColors.neutral.mediumGray,
  },
};

// ============================================
// DROPDOWN / SELECT COMPONENT
// ============================================
export const dropdownSpec = {
  trigger: inputSpec,
  menu: {
    marginTop: 4,
    borderRadius: 8,
    background: eyColors.neutral.white,
    shadow: '0 8px 24px rgba(15, 15, 15, 0.15)',
    border: `1px solid ${eySemanticColors.border.default}`,
    maxHeight: 300,
    overflow: 'auto',
  },
  item: {
    height: 40,
    padding: { horizontal: 12 },
    fontSize: 14,
    states: {
      default: {
        background: 'transparent',
        color: eyColors.neutral.darkGray,
      },
      hover: {
        background: eyColors.neutral.lightGray,
      },
      selected: {
        background: eyColors.primary.yellow + '20',
        fontWeight: 500,
      },
    },
  },
};

// ============================================
// TOOLTIP COMPONENT
// ============================================
export const tooltipSpec = {
  maxWidth: 300,
  padding: { horizontal: 12, vertical: 8 },
  borderRadius: 6,
  background: eyColors.neutral.darkGray,
  color: eyColors.neutral.white,
  fontSize: 12,
  lineHeight: '16px',
  shadow: '0 4px 12px rgba(0, 0, 0, 0.2)',
  arrow: {
    size: 6,
  },
};

// ============================================
// TOAST / NOTIFICATION COMPONENT
// ============================================
export const toastSpec = {
  minWidth: 320,
  maxWidth: 480,
  padding: 16,
  borderRadius: 8,
  gap: 12,
  shadow: '0 8px 24px rgba(15, 15, 15, 0.15)',
  variants: {
    success: {
      background: eyColors.neutral.white,
      borderLeft: `4px solid ${eyColors.status.success}`,
      iconColor: eyColors.status.success,
    },
    warning: {
      background: eyColors.neutral.white,
      borderLeft: `4px solid ${eyColors.status.warning}`,
      iconColor: eyColors.status.warning,
    },
    error: {
      background: eyColors.neutral.white,
      borderLeft: `4px solid ${eyColors.status.critical}`,
      iconColor: eyColors.status.critical,
    },
    info: {
      background: eyColors.neutral.white,
      borderLeft: `4px solid ${eyColors.status.info}`,
      iconColor: eyColors.status.info,
    },
  },
};

export type ComponentSpec =
  | typeof headerSpec
  | typeof leftNavSpec
  | typeof kpiTileSpec
  | typeof tableSpec
  | typeof buttonSpec
  | typeof badgeSpec
  | typeof tabSpec
  | typeof modalSpec
  | typeof progressSpec
  | typeof chartSpec
  | typeof cardSpec
  | typeof inputSpec
  | typeof dropdownSpec
  | typeof tooltipSpec
  | typeof toastSpec;
