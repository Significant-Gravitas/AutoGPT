import type { Platform, WorkloadFormValues } from "../../helpers";

export interface FieldConfig {
  name: keyof WorkloadFormValues;
  label: string;
  placeholder: string;
  hint?: string;
}

export const PLATFORM_FIELDS: Record<Platform, FieldConfig[]> = {
  github: [
    { name: "repository", label: "Repository", placeholder: "acme/api" },
    {
      name: "branch",
      label: "Branch (optional)",
      placeholder: "main",
      hint: "Leave empty to trust workflows on every branch.",
    },
    {
      name: "environment",
      label: "GitHub environment (optional)",
      placeholder: "production",
    },
  ],
  vercel: [
    {
      name: "team",
      label: "Team slug",
      placeholder: "acme",
      hint: "The name in your team's URL, vercel.com/acme.",
    },
    { name: "project", label: "Project name", placeholder: "acme-website" },
  ],
  gitlab: [
    { name: "projectPath", label: "Project path", placeholder: "acme/api" },
    {
      name: "branch",
      label: "Branch (optional)",
      placeholder: "main",
      hint: "Leave empty to trust pipelines on every branch.",
    },
  ],
  google: [
    {
      name: "serviceAccount",
      label: "Service account email",
      placeholder: "agent@my-project.iam.gserviceaccount.com",
    },
  ],
  website: [
    {
      name: "siteUrl",
      label: "Your site",
      placeholder: "https://acme.com",
      hint: "Your server will publish its public key here.",
    },
    {
      name: "subject",
      label: "Server name",
      placeholder: "backend",
      hint: "Your server puts this name in every token it signs.",
    },
  ],
  custom: [
    {
      name: "issuer",
      label: "Issuer URL",
      placeholder: "https://login.example.com",
    },
    {
      name: "audience",
      label: "Audience (optional)",
      placeholder: "Defaults to AutoGPT's address",
    },
    {
      name: "claim",
      label: "Claim that names your workload",
      placeholder: "sub",
    },
    {
      name: "pattern",
      label: "Value it must have",
      placeholder: "system:serviceaccount:prod:agent",
      hint: "Exact, or with * as a wildcard after a fixed start.",
    },
  ],
};

export const VERCEL_ENVIRONMENTS = [
  { value: "production", label: "Production" },
  { value: "preview", label: "Preview" },
  { value: "development", label: "Development" },
  { value: "any", label: "Any environment" },
];

export const VERCEL_ISSUER_MODES = [
  { value: "team", label: "Team (Vercel's default)" },
  { value: "global", label: "Global" },
];
