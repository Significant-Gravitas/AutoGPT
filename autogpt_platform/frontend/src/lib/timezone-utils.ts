/**
 * Utility functions for timezone conversions and display
 */

/**
 * Format a date/time in the user's timezone with timezone indicator
 * @param date - The date to format (can be string or Date)
 * @param timezone - The IANA timezone identifier (e.g., "America/New_York")
 * @param options - Intl.DateTimeFormat options
 * @returns Formatted date string with timezone
 */
export function formatInTimezone(
  date: string | Date,
  timezone: string,
  options?: Intl.DateTimeFormatOptions,
): string {
  const dateObj = typeof date === "string" ? new Date(date) : date;

  const defaultOptions: Intl.DateTimeFormatOptions = {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    ...options,
  };

  try {
    return new Intl.DateTimeFormat("en-US", {
      ...defaultOptions,
      timeZone: timezone === "not-set" ? undefined : timezone,
    }).format(dateObj);
  } catch {
    // Fallback to local timezone if invalid timezone
    console.warn(`Invalid timezone "${timezone}", using local timezone`);
    return new Intl.DateTimeFormat("en-US", defaultOptions).format(dateObj);
  }
}

/**
 * Get the timezone abbreviation (e.g., "EST", "PST")
 * @param timezone - The IANA timezone identifier
 * @returns Timezone abbreviation
 */
export function getTimezoneAbbreviation(timezone: string): string {
  if (timezone === "not-set" || !timezone) {
    return "";
  }

  try {
    const date = new Date();
    const tzAbbr = new Intl.DateTimeFormat("en-US", {
      timeZone: timezone,
      timeZoneName: "short",
    })
      .formatToParts(date)
      .find((part) => part.type === "timeZoneName")?.value;

    return tzAbbr || timezone;
  } catch {
    return timezone;
  }
}

/**
 * Get a human-readable timezone name
 * @param timezone - IANA timezone identifier
 * @returns Human-readable name
 */
export function getTimezoneDisplayName(timezone: string): string {
  if (timezone === "not-set") {
    return "Not set";
  }

  if (timezone === "UTC") {
    return "UTC";
  }

  // Convert America/New_York to "New York (EST)"
  try {
    const parts = timezone.split("/");
    const city = parts[parts.length - 1].replace(/_/g, " ");
    const abbr = getTimezoneAbbreviation(timezone);
    if (abbr && abbr !== timezone) {
      return `${city} (${abbr})`;
    }
    // If abbreviation is same as timezone or not found, show timezone with underscores replaced
    const timezoneDisplay = timezone.replace(/_/g, " ");
    return `${city} (${timezoneDisplay})`;
  } catch {
    return timezone.replace(/_/g, " ");
  }
}
