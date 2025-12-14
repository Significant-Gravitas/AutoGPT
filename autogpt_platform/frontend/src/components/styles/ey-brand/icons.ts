/**
 * EY Brand Icon Set
 *
 * 24px icons with filled and outlined variants
 * Stroke width: 1.5-2px for consistency
 *
 * Usage:
 * - Import as SVG from /public/ey-brand/icons/
 * - Or use the icon names with your icon component library
 */

export const eyIcons = {
  // Core navigation icons
  notifications: {
    outlined: '/ey-brand/icons/notifications-outlined.svg',
    filled: '/ey-brand/icons/notifications-filled.svg',
  },
  search: {
    outlined: '/ey-brand/icons/search-outlined.svg',
    filled: '/ey-brand/icons/search-filled.svg',
  },
  filter: {
    outlined: '/ey-brand/icons/filter-outlined.svg',
    filled: '/ey-brand/icons/filter-filled.svg',
  },

  // Action icons
  download: {
    outlined: '/ey-brand/icons/download-outlined.svg',
    filled: '/ey-brand/icons/download-filled.svg',
  },
  edit: {
    outlined: '/ey-brand/icons/edit-outlined.svg',
    filled: '/ey-brand/icons/edit-filled.svg',
  },
  mail: {
    outlined: '/ey-brand/icons/mail-outlined.svg',
    filled: '/ey-brand/icons/mail-filled.svg',
  },
  print: {
    outlined: '/ey-brand/icons/print-outlined.svg',
    filled: '/ey-brand/icons/print-filled.svg',
  },
  csv: {
    outlined: '/ey-brand/icons/csv-outlined.svg',
    filled: '/ey-brand/icons/csv-filled.svg',
  },
  calendar: {
    outlined: '/ey-brand/icons/calendar-outlined.svg',
    filled: '/ey-brand/icons/calendar-filled.svg',
  },

  // Status icons
  warning: {
    outlined: '/ey-brand/icons/warning-outlined.svg',
    filled: '/ey-brand/icons/warning-filled.svg',
  },
  check: {
    outlined: '/ey-brand/icons/check-outlined.svg',
    filled: '/ey-brand/icons/check-filled.svg',
  },
  info: {
    outlined: '/ey-brand/icons/info-outlined.svg',
    filled: '/ey-brand/icons/info-filled.svg',
  },

  // Domain-specific icons
  pod: {
    outlined: '/ey-brand/icons/pod-outlined.svg',
    filled: '/ey-brand/icons/pod-filled.svg',
  },
} as const;

export type EYIconName = keyof typeof eyIcons;
export type EYIconVariant = 'outlined' | 'filled';

/**
 * Get the path to an EY icon
 */
export function getEYIconPath(name: EYIconName, variant: EYIconVariant = 'outlined'): string {
  return eyIcons[name][variant];
}

/**
 * All available icon names for documentation
 */
export const eyIconNames: EYIconName[] = Object.keys(eyIcons) as EYIconName[];
