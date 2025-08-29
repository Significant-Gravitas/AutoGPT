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

export function getMissingRequiredInputs(
  inputSchema: any,
  values: Record<string, any>,
): string[] {
  if (!inputSchema || typeof inputSchema !== "object") return [];
  const required: string[] = Array.isArray(inputSchema.required)
    ? inputSchema.required
    : [];
  const properties: Record<string, any> = inputSchema.properties || {};
  return required.filter((key) => {
    const field = properties[key];
    if (field?.hidden) return false;
    return isEmpty(values[key]);
  });
}

export function getMissingCredentials(
  credentialsProperties: Record<string, any> | undefined,
  values: Record<string, any>,
): string[] {
  const props = credentialsProperties || {};
  return Object.keys(props).filter((key) => !(key in values));
}

type DeriveReadinessParams = {
  inputSchema: any;
  credentialsProperties?: Record<string, any>;
  values: Record<string, any>;
  credentialsValues: Record<string, any>;
};

export function deriveReadiness(params: DeriveReadinessParams): {
  missingInputs: string[];
  missingCredentials: string[];
  credentialsRequired: boolean;
  allRequiredInputsAreSet: boolean;
} {
  const missingInputs = getMissingRequiredInputs(
    params.inputSchema,
    params.values,
  );
  const credentialsRequired =
    Object.keys(params.credentialsProperties || {}).length > 0;
  const missingCredentials = getMissingCredentials(
    params.credentialsProperties,
    params.credentialsValues,
  );
  const allRequiredInputsAreSet =
    missingInputs.length === 0 &&
    (!credentialsRequired || missingCredentials.length === 0);
  return {
    missingInputs,
    missingCredentials,
    credentialsRequired,
    allRequiredInputsAreSet,
  };
}

export function getVisibleInputFields(inputSchema: any): Record<string, any> {
  if (
    !inputSchema ||
    typeof inputSchema !== "object" ||
    !("properties" in inputSchema) ||
    !inputSchema.properties
  ) {
    return {} as Record<string, any>;
  }
  const properties = inputSchema.properties as Record<string, any>;
  return Object.fromEntries(
    Object.entries(properties).filter(([, subSchema]) => !subSchema?.hidden),
  );
}

export function getCredentialFields(
  credentialsInputSchema: any,
): Record<string, any> {
  if (
    !credentialsInputSchema ||
    typeof credentialsInputSchema !== "object" ||
    !("properties" in credentialsInputSchema) ||
    !credentialsInputSchema.properties
  ) {
    return {} as Record<string, any>;
  }
  return credentialsInputSchema.properties as Record<string, any>;
}

type CollectMissingFieldsOptions = {
  needScheduleName?: boolean;
  scheduleName: string;
  missingInputs: string[];
  credentialsRequired: boolean;
  allCredentialsAreSet: boolean;
  missingCredentials: string[];
};

export function collectMissingFields(
  options: CollectMissingFieldsOptions,
): string[] {
  const scheduleMissing =
    options.needScheduleName && !options.scheduleName ? ["schedule_name"] : [];

  const missingCreds =
    options.credentialsRequired && !options.allCredentialsAreSet
      ? options.missingCredentials.map((k) => `credentials:${k}`)
      : [];

  return ([] as string[])
    .concat(scheduleMissing)
    .concat(options.missingInputs)
    .concat(missingCreds);
}

export function getErrorMessage(error: unknown): string {
  if (typeof error === "string") return error;
  if (error && typeof error === "object" && "message" in error) {
    const msg = (error as any).message;
    if (typeof msg === "string" && msg.trim().length > 0) return msg;
  }
  return "An unexpected error occurred.";
}
