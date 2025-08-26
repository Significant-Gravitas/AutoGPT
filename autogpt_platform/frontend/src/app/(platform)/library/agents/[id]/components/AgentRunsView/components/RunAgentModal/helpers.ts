import { CredentialsMetaInput } from "@/app/api/__generated__/models/credentialsMetaInput";
import { isEmpty } from "@/lib/utils";

export function validateInputs(
  inputSchema: any,
  values: Record<string, any>,
): Record<string, string> {
  const errors: Record<string, string> = {};

  if (!inputSchema?.properties) return errors;

  const requiredFields = inputSchema.required || [];

  for (const fieldName of requiredFields) {
    const fieldSchema = inputSchema.properties[fieldName];
    if (!fieldSchema?.hidden && isEmpty(values[fieldName])) {
      errors[fieldName] = `${fieldSchema?.title || fieldName} is required`;
    }
  }

  return errors;
}

export function validateCredentials(
  credentialsSchema: any,
  values: Record<string, CredentialsMetaInput>,
): Record<string, string> {
  const errors: Record<string, string> = {};

  if (!credentialsSchema?.properties) return errors;

  const credentialFields = Object.keys(credentialsSchema.properties);

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
