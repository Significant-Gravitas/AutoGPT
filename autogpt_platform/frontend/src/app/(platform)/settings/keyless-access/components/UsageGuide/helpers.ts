import type { WorkloadIdentityInfo } from "@/app/api/__generated__/models/workloadIdentityInfo";

import { exampleEndpoint, type Platform } from "../../helpers";

export interface Snippet {
  label: string;
  code: string;
}

export interface TokenStep {
  intro: string;
  snippets: Snippet[];
}

interface Args {
  identity: WorkloadIdentityInfo;
  platform: Platform;
}

export function tokenStep({ identity, platform }: Args): TokenStep {
  const audience = identity.audience;
  switch (platform) {
    case "github":
      return {
        intro:
          "Let the job ask GitHub for an identity token, then request one for AutoGPT in a run step.",
        snippets: [
          {
            label: "Workflow file",
            code: `permissions:\n  id-token: write\n  contents: read`,
          },
          {
            label: "Run step",
            code: `OIDC_TOKEN=$(curl -s \\\n  -H "Authorization: Bearer $ACTIONS_ID_TOKEN_REQUEST_TOKEN" \\\n  "$ACTIONS_ID_TOKEN_REQUEST_URL&audience=${audience}" | jq -r .value)`,
          },
        ],
      };
    case "gitlab":
      return {
        intro:
          "Ask GitLab to put an identity token for AutoGPT into the job's environment.",
        snippets: [
          {
            label: ".gitlab-ci.yml",
            code: `call-autogpt:\n  id_tokens:\n    AUTOGPT_ID_TOKEN:\n      aud: ${audience}\n  script:\n    - OIDC_TOKEN="$AUTOGPT_ID_TOKEN"`,
          },
        ],
      };
    case "google":
      return {
        intro:
          "On a VM, Cloud Run or GKE running as the service account, ask the metadata server. format=full adds the email AutoGPT checks.",
        snippets: [
          {
            label: "Terminal",
            code: `OIDC_TOKEN=$(curl -s -H "Metadata-Flavor: Google" \\\n  "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/identity?audience=${audience}&format=full")`,
          },
        ],
      };
    case "vercel":
      return {
        intro:
          "Inside a Vercel Function, @vercel/oidc returns a token for your project. Ask for one meant for AutoGPT.",
        snippets: [
          {
            label: "Function code",
            code: `import { getVercelOidcToken } from "@vercel/oidc";\n\nconst oidcToken = await getVercelOidcToken({\n  audience: "${audience}",\n});`,
          },
        ],
      };
    case "website":
    case "custom":
      return {
        intro: `Get a token signed by ${identity.issuer} for the audience ${audience}. If that is your own site, your server signs it with its private key:`,
        snippets: [
          {
            label: "Node.js (npm install jose)",
            code: signSnippet(identity),
          },
        ],
      };
  }
}

export function exchangeSnippet(identityId: string, baseUrl: string) {
  return [
    `WORKLOAD_ID="${identityId}"`,
    "",
    `AUTOGPT_TOKEN=$(curl -s ${baseUrl}/api/oauth/token \\`,
    `  -H "Content-Type: application/json" \\`,
    `  -d "{\\"grant_type\\": \\"urn:ietf:params:oauth:grant-type:token-exchange\\",`,
    `       \\"subject_token_type\\": \\"urn:ietf:params:oauth:token-type:id_token\\",`,
    `       \\"client_id\\": \\"$WORKLOAD_ID\\",`,
    `       \\"subject_token\\": \\"$OIDC_TOKEN\\"}" \\`,
    `  | jq -r .access_token)`,
  ].join("\n");
}

export function callSnippet(identity: WorkloadIdentityInfo, baseUrl: string) {
  const url = `${baseUrl}${exampleEndpoint(identity.permissions)}`;
  return `curl ${url} \\\n  -H "Authorization: Bearer $AUTOGPT_TOKEN"`;
}

export function fetchSnippet(identity: WorkloadIdentityInfo, baseUrl: string) {
  const url = `${baseUrl}${exampleEndpoint(identity.permissions)}`;
  return [
    `const exchange = await fetch("${baseUrl}/api/oauth/token", {`,
    '  method: "POST",',
    '  headers: { "Content-Type": "application/json" },',
    "  body: JSON.stringify({",
    '    grant_type: "urn:ietf:params:oauth:grant-type:token-exchange",',
    '    subject_token_type: "urn:ietf:params:oauth:token-type:id_token",',
    `    client_id: "${identity.id}",`,
    "    subject_token: oidcToken,",
    "  }),",
    "});",
    "const { access_token } = await exchange.json();",
    "",
    `const response = await fetch("${url}", {`,
    "  headers: { Authorization: `Bearer ${access_token}` },",
    "});",
  ].join("\n");
}

export function apiKeyCallSnippet(
  identity: WorkloadIdentityInfo,
  baseUrl: string,
) {
  const url = `${baseUrl}${exampleEndpoint(identity.permissions)}`;
  return `curl ${url} \\\n  -H "X-API-Key: $AUTOGPT_API_KEY"`;
}

export function siteSetupSnippets(identity: WorkloadIdentityInfo) {
  const issuer = identity.issuer;
  return {
    keygen: [
      'import { exportJWK, exportPKCS8, generateKeyPair } from "jose";',
      "",
      'const { publicKey, privateKey } = await generateKeyPair("ES256", {',
      "  extractable: true,",
      "});",
      "const jwk = await exportJWK(publicKey);",
      "",
      "// Keep this on your server only, e.g. as AUTOGPT_SIGNING_KEY",
      "console.log(await exportPKCS8(privateKey));",
      "// Publish this as /.well-known/jwks.json",
      "console.log(",
      "  JSON.stringify({",
      '    keys: [{ ...jwk, kid: "autogpt-1", alg: "ES256", use: "sig" }],',
      "  }),",
      ");",
    ].join("\n"),
    discovery: JSON.stringify(
      { issuer, jwks_uri: `${issuer}/.well-known/jwks.json` },
      null,
      2,
    ),
    discoveryUrl: `${issuer}/.well-known/openid-configuration`,
  };
}

function signSnippet(identity: WorkloadIdentityInfo) {
  const subject = subjectOf(identity);
  return [
    'import { importPKCS8, SignJWT } from "jose";',
    "",
    'const key = await importPKCS8(process.env.AUTOGPT_SIGNING_KEY, "ES256");',
    "const oidcToken = await new SignJWT({})",
    '  .setProtectedHeader({ alg: "ES256", kid: "autogpt-1" })',
    `  .setIssuer("${identity.issuer}")`,
    `  .setSubject("${subject}")`,
    `  .setAudience("${identity.audience}")`,
    "  .setIssuedAt()",
    '  .setExpirationTime("5m")',
    "  .sign(key);",
  ].join("\n");
}

function subjectOf(identity: WorkloadIdentityInfo) {
  const condition = identity.claim_conditions.find((c) => c.claim === "sub");
  return condition?.pattern ?? "your-server";
}
