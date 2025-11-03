/**
 * Cookie consent preferences for GDPR compliance
 */
export interface ConsentPreferences {
  /** Whether the user has made a consent choice */
  hasConsented: boolean;
  /** Timestamp when consent was given/updated */
  timestamp: number;
  /** Analytics cookies (Google Analytics, Vercel Analytics, Datafa.st) */
  analytics: boolean;
  /** Error monitoring and session replay (Sentry) */
  monitoring: boolean;
}

/**
 * Default consent state - all non-essential cookies disabled
 */
export const DEFAULT_CONSENT: ConsentPreferences = {
  hasConsented: false,
  timestamp: Date.now(),
  analytics: false,
  monitoring: false,
};

/**
 * Cookie categories with descriptions
 */
export const COOKIE_CATEGORIES = {
  essential: {
    name: "Essential Cookies",
    description: "Required for login, authentication, and core functionality",
    alwaysActive: true,
  },
  analytics: {
    name: "Analytics & Performance",
    description:
      "Help us understand how you use AutoGPT to improve our service (Google Analytics, Vercel Analytics, Datafa.st)",
    alwaysActive: false,
  },
  monitoring: {
    name: "Error Monitoring & Session Replay",
    description:
      "Record errors and user sessions to help us fix bugs faster (Sentry - includes screen recording)",
    alwaysActive: false,
  },
} as const;
