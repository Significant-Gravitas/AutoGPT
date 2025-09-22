import { CredentialsMetaInput } from "@/app/api/__generated__/models/credentialsMetaInput";
import { isEmpty } from "@/lib/utils";

export function validateInputs(
  inputSchema: unknown,
  values: Record<string, unknown>,
): Record<string, string> {
  const errors: Record<string, string> = {};

  if (
    !(
      inputSchema &&
      typeof inputSchema === "object" &&
      (inputSchema as any).properties
    )
  )
    return errors;

  const requiredFields = ((inputSchema as any).required || []) as string[];

  for (const fieldName of requiredFields) {
    const fieldSchema = (inputSchema as any).properties[fieldName];
    if (!fieldSchema?.hidden && isEmpty((values as any)[fieldName])) {
      errors[fieldName] =
        `${(fieldSchema as any)?.title || fieldName} is required`;
    }
  }

  return errors;
}

export function validateCredentials(
  credentialsSchema: unknown,
  values: Record<string, CredentialsMetaInput>,
): Record<string, string> {
  const errors: Record<string, string> = {};

  if (
    !(
      credentialsSchema &&
      typeof credentialsSchema === "object" &&
      (credentialsSchema as any).properties
    )
  )
    return errors;

  const credentialFields = Object.keys((credentialsSchema as any).properties);

  for (const fieldName of credentialFields) {
    if (!values[fieldName]) {
      errors[fieldName] = `${fieldName} credentials are required`;
    }
  }

  return errors;
}

export function formatCronExpression(cron: string): string {
  // Basic cron expression formatting/validation
  const parts = cron.trim().split(/\s+/);
  if (parts.length !== 5) {
    throw new Error(
      "Cron expression must have exactly 5 parts: minute hour day month weekday",
    );
  }
  return parts.join(" ");
}

export function parseCronDescription(cron: string): string {
  // Simple cron description parser
  const parts = cron.split(" ");
  if (parts.length !== 5) return cron;

  // Handle some common patterns
  if (cron === "0 * * * *") return "Every hour";
  if (cron === "0 9 * * *") return "Daily at 9:00 AM";
  if (cron === "0 9 * * 1") return "Every Monday at 9:00 AM";
  if (cron === "0 9 * * 1-5") return "Weekdays at 9:00 AM";
  if (cron === "0 9 1 * *") return "Monthly on the 1st at 9:00 AM";

  return cron; // Fallback to showing the raw cron
}
