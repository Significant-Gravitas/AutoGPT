import {
  CloudServerIcon,
  GithubIcon,
  GitlabIcon,
  Globe02Icon,
  GoogleIcon,
  ShieldKeyIcon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";

import { APIKeyPermission } from "@/app/api/__generated__/models/aPIKeyPermission";
import type { ClaimCondition } from "@/app/api/__generated__/models/claimCondition";
import type { WorkloadIdentityInfo } from "@/app/api/__generated__/models/workloadIdentityInfo";
import type { WorkloadIdentitySpec } from "@/app/api/__generated__/models/workloadIdentitySpec";

export type Platform =
  | "github"
  | "vercel"
  | "gitlab"
  | "google"
  | "website"
  | "custom";

export const GITHUB_ISSUER = "https://token.actions.githubusercontent.com";
export const GITLAB_ISSUER = "https://gitlab.com";
export const GOOGLE_ISSUER = "https://accounts.google.com";
export const VERCEL_ISSUER = "https://oidc.vercel.com";

export const KEYLESS_DOCS_URL =
  "https://agpt.co/docs/platform/integrating/keyless-access";

export interface PlatformOption {
  value: Platform;
  label: string;
  description: string;
  icon: IconSvgElement;
}

export const PLATFORM_OPTIONS: PlatformOption[] = [
  {
    value: "github",
    label: "GitHub Actions",
    description: "Workflows in one of your repositories",
    icon: GithubIcon,
  },
  {
    value: "vercel",
    label: "Vercel",
    description: "Functions in one of your projects",
    icon: CloudServerIcon,
  },
  {
    value: "gitlab",
    label: "GitLab CI",
    description: "Pipelines in one gitlab.com project",
    icon: GitlabIcon,
  },
  {
    value: "google",
    label: "Google Cloud",
    description: "Code running as one service account",
    icon: GoogleIcon,
  },
  {
    value: "website",
    label: "Your own website or server",
    description: "Any server that can publish a public key on your domain",
    icon: Globe02Icon,
  },
  {
    value: "custom",
    label: "Another OIDC provider",
    description: "Kubernetes, Auth0, self-managed GitLab and others",
    icon: ShieldKeyIcon,
  },
];

// A type alias, not an interface: react-hook-form's FieldValues needs the
// implicit index signature that only aliases get.
export type WorkloadFormValues = {
  platform: Platform;
  name: string;
  permissions: APIKeyPermission[];
  repository: string;
  branch: string;
  environment: string;
  team: string;
  project: string;
  vercelEnvironment: string;
  vercelIssuerMode: string;
  projectPath: string;
  serviceAccount: string;
  siteUrl: string;
  subject: string;
  issuer: string;
  audience: string;
  claim: string;
  pattern: string;
};

export const EMPTY_FORM_VALUES: WorkloadFormValues = {
  platform: "github",
  name: "",
  permissions: [],
  repository: "",
  branch: "",
  environment: "",
  team: "",
  project: "",
  vercelEnvironment: "production",
  vercelIssuerMode: "team",
  projectPath: "",
  serviceAccount: "",
  siteUrl: "",
  subject: "",
  issuer: "",
  audience: "",
  claim: "sub",
  pattern: "",
};

interface Trust {
  issuer: string;
  audience: string;
  conditions: ClaimCondition[];
}

export function buildWorkloadSpec(
  values: WorkloadFormValues,
  audience: string,
): WorkloadIdentitySpec {
  const trust = trustFor(values, audience);
  return {
    name: values.name.trim() || defaultName(values),
    issuer: trust.issuer,
    audience: trust.audience,
    claim_conditions: trust.conditions,
    permissions: values.permissions,
  };
}

export function platformOf(identity: Pick<WorkloadIdentityInfo, "issuer">) {
  const { issuer } = identity;
  if (issuer === GITHUB_ISSUER) return "github";
  if (issuer === VERCEL_ISSUER || issuer.startsWith(`${VERCEL_ISSUER}/`)) {
    return "vercel";
  }
  if (issuer === GITLAB_ISSUER) return "gitlab";
  if (issuer === GOOGLE_ISSUER) return "google";
  return "custom";
}

export function platformOption(platform: Platform): PlatformOption {
  const found = PLATFORM_OPTIONS.find((option) => option.value === platform);
  return found ?? PLATFORM_OPTIONS[PLATFORM_OPTIONS.length - 1];
}

export function describeTrust(conditions: ClaimCondition[]) {
  return conditions.map((c) => `${c.claim} = ${c.pattern}`).join(" · ");
}

export function exampleEndpoint(permissions: APIKeyPermission[]) {
  if (permissions.includes(APIKeyPermission.IDENTITY)) {
    return "/external-api/v1/me";
  }
  if (permissions.includes(APIKeyPermission.READ_BLOCK)) {
    return "/external-api/v1/blocks";
  }
  if (permissions.includes(APIKeyPermission.READ_STORE)) {
    return "/external-api/v1/store/agents";
  }
  return "/external-api/v1/me";
}

export function normalizeSiteUrl(raw: string) {
  const url = new URL(raw.trim());
  return `${url.origin}${url.pathname.replace(/\/+$/, "")}`;
}

function trustFor(values: WorkloadFormValues, audience: string): Trust {
  switch (values.platform) {
    case "github":
      return {
        issuer: GITHUB_ISSUER,
        audience,
        conditions: conditions([
          ["repository", values.repository],
          ["ref", values.branch && `refs/heads/${values.branch.trim()}`],
          ["environment", values.environment],
        ]),
      };
    case "vercel":
      return {
        issuer:
          values.vercelIssuerMode === "global"
            ? VERCEL_ISSUER
            : `${VERCEL_ISSUER}/${values.team.trim()}`,
        audience,
        conditions: conditions([
          ["owner", values.team],
          ["project", values.project],
          [
            "environment",
            values.vercelEnvironment === "any" ? "" : values.vercelEnvironment,
          ],
        ]),
      };
    case "gitlab":
      return {
        issuer: GITLAB_ISSUER,
        audience,
        conditions: conditions([
          ["project_path", values.projectPath],
          ["ref_type", values.branch && "branch"],
          ["ref", values.branch],
        ]),
      };
    case "google":
      return {
        issuer: GOOGLE_ISSUER,
        audience,
        conditions: conditions([["email", values.serviceAccount]]),
      };
    case "website":
      return {
        issuer: normalizeSiteUrl(values.siteUrl),
        audience,
        conditions: conditions([["sub", values.subject]]),
      };
    case "custom":
      return {
        issuer: values.issuer.trim(),
        audience: values.audience.trim() || audience,
        conditions: conditions([[values.claim, values.pattern]]),
      };
  }
}

function conditions(pairs: [string, string][]): ClaimCondition[] {
  return pairs
    .map(([claim, pattern]) => ({
      claim: claim.trim(),
      pattern: pattern.trim(),
    }))
    .filter((condition) => condition.claim && condition.pattern);
}

function defaultName(values: WorkloadFormValues) {
  switch (values.platform) {
    case "github":
      return `GitHub Actions · ${values.repository.trim()}`;
    case "vercel":
      return `Vercel · ${values.project.trim()}`;
    case "gitlab":
      return `GitLab CI · ${values.projectPath.trim()}`;
    case "google":
      return `Google Cloud · ${values.serviceAccount.trim()}`;
    case "website":
      return `${new URL(values.siteUrl.trim()).host} · ${values.subject.trim()}`;
    case "custom":
      return new URL(values.issuer.trim()).host;
  }
}
