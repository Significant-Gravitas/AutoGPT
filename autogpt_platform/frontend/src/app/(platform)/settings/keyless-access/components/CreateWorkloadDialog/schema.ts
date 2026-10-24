import { z } from "zod";

import { APIKeyPermission } from "@/app/api/__generated__/models/aPIKeyPermission";

import type { WorkloadFormValues } from "../../helpers";

const REPOSITORY = /^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/;
const GITLAB_PROJECT = /^[A-Za-z0-9_.-]+(\/[A-Za-z0-9_.-]+)+$/;
const VERCEL_NAME = /^[a-z0-9][a-z0-9._-]*$/;
const LITERAL = /^[^\s*?[]+$/;
const SUBJECT = /^[A-Za-z0-9_.:@/-]+$/;
const EMAIL = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
const VERIFIED_CLAIMS = ["iss", "aud", "exp", "iat", "nbf"];

type Field = keyof WorkloadFormValues;
type Problem = [Field, string];

export const workloadFormSchema = z
  .object({
    platform: z.enum([
      "github",
      "vercel",
      "gitlab",
      "google",
      "website",
      "custom",
    ]),
    name: z.string().trim().max(100, "Name must be 100 characters or less"),
    permissions: z
      .array(z.nativeEnum(APIKeyPermission))
      .min(1, "Select at least one permission"),
    repository: z.string().trim(),
    branch: z.string().trim(),
    environment: z.string().trim(),
    team: z.string().trim(),
    project: z.string().trim(),
    vercelEnvironment: z.string(),
    vercelIssuerMode: z.string(),
    projectPath: z.string().trim(),
    serviceAccount: z.string().trim(),
    siteUrl: z.string().trim(),
    subject: z.string().trim(),
    issuer: z.string().trim(),
    audience: z.string().trim(),
    claim: z.string().trim(),
    pattern: z.string().trim(),
  })
  .superRefine((values, ctx) => {
    for (const [field, message] of problemsWith(values)) {
      ctx.addIssue({ code: z.ZodIssueCode.custom, path: [field], message });
    }
  });

export function problemsWith(values: WorkloadFormValues): Problem[] {
  switch (values.platform) {
    case "github":
      return [
        ...required(values, "repository", REPOSITORY, "Use owner/repository"),
        ...optional(values, "branch", LITERAL, "Use one branch name"),
        ...optional(values, "environment", LITERAL, "Use one environment"),
      ];
    case "vercel":
      return [
        ...required(values, "team", VERCEL_NAME, "Use your team's URL slug"),
        ...required(values, "project", VERCEL_NAME, "Use the project's name"),
      ];
    case "gitlab":
      return [
        ...required(values, "projectPath", GITLAB_PROJECT, "Use group/project"),
        ...optional(values, "branch", LITERAL, "Use one branch name"),
      ];
    case "google":
      return required(
        values,
        "serviceAccount",
        EMAIL,
        "Use the account's email",
      );
    case "website":
      return [
        ...httpsUrl(values, "siteUrl"),
        ...required(
          values,
          "subject",
          SUBJECT,
          "Letters, digits and . _ : @ / -",
        ),
      ];
    case "custom":
      return [...httpsUrl(values, "issuer"), ...customCondition(values)];
  }
}

function required(
  values: WorkloadFormValues,
  field: Field,
  pattern: RegExp,
  message: string,
): Problem[] {
  const value = String(values[field]).trim();
  if (!value) return [[field, "Required"]];
  return pattern.test(value) ? [] : [[field, message]];
}

function optional(
  values: WorkloadFormValues,
  field: Field,
  pattern: RegExp,
  message: string,
): Problem[] {
  const value = String(values[field]).trim();
  return !value || pattern.test(value) ? [] : [[field, message]];
}

function httpsUrl(values: WorkloadFormValues, field: Field): Problem[] {
  const value = String(values[field]).trim();
  if (!value) return [[field, "Required"]];
  try {
    const url = new URL(value);
    const valid =
      url.protocol === "https:" &&
      !url.username &&
      !url.password &&
      !url.search &&
      !url.hash;
    return valid ? [] : [[field, "Use an https:// address with no query"]];
  } catch {
    return [[field, "Use an https:// address"]];
  }
}

function customCondition(values: WorkloadFormValues): Problem[] {
  const problems: Problem[] = [];
  if (!values.claim.trim()) problems.push(["claim", "Required"]);
  if (VERIFIED_CLAIMS.includes(values.claim.trim())) {
    problems.push(["claim", "Pick a claim that names your workload"]);
  }
  if (!values.pattern.trim()) problems.push(["pattern", "Required"]);
  if (/^[*?[]/.test(values.pattern.trim())) {
    problems.push(["pattern", "Start with a literal value, not a wildcard"]);
  }
  return problems;
}
