import { http, HttpResponse } from "msw";
import type { CredentialsMetaResponse } from "@/app/api/__generated__/models/credentialsMetaResponse";

export const storyCredentials: CredentialsMetaResponse[] = [
  {
    id: "google-work",
    provider: "google",
    type: "oauth2",
    title: "Work Gmail",
    username: "alex@company.com",
    scopes: null,
  },
  {
    id: "google-personal",
    provider: "google",
    type: "oauth2",
    title: "Personal Gmail",
    username: "alex@gmail.com",
    scopes: null,
  },
];

const storyProviders = [
  {
    name: "linear",
    display_name: "Linear",
    description: "Issues, projects and cycles",
    supported_auth_types: ["oauth2", "api_key"],
  },
  {
    name: "github",
    display_name: "GitHub",
    description: "Repositories, issues and pull requests",
    supported_auth_types: ["oauth2"],
  },
  {
    name: "notion",
    display_name: "Notion",
    description: "Docs and databases",
    supported_auth_types: ["oauth2"],
  },
];

export const mentionStoryHandlers = [
  http.get("*/api/integrations/credentials", () =>
    HttpResponse.json(storyCredentials),
  ),
  http.get("*/api/integrations/providers", () =>
    HttpResponse.json(storyProviders),
  ),
  http.get("*/api/experts/:expertId/credentials", () =>
    HttpResponse.json([
      {
        credential_id: "google-work",
        provider: "google",
        title: "Work Gmail",
        type: "oauth2",
      },
    ]),
  ),
  http.get("*/api/workspace/files", ({ request }) => {
    const query =
      new URL(request.url).searchParams.get("q")?.toLowerCase() ?? "";
    return HttpResponse.json({
      files: [
        {
          id: "file-todos",
          name: "Team TODOs.md",
          path: "/workspace/Team TODOs.md",
          mime_type: "text/markdown",
          size_bytes: 1240,
          origin: "uploaded",
          created_at: "2026-09-24T10:00:00Z",
        },
        {
          id: "file-notes",
          name: "Meeting notes.pdf",
          path: "/workspace/Meeting notes.pdf",
          mime_type: "application/pdf",
          size_bytes: 42000,
          origin: "uploaded",
          created_at: "2026-09-24T10:00:00Z",
        },
      ].filter((file) => file.name.toLowerCase().includes(query)),
      has_more: false,
    });
  }),
  http.get("*/api/workspace/folders", () =>
    HttpResponse.json({
      folders: [
        {
          id: "folder-project",
          workspace_id: "workspace",
          name: "Project files",
          parent_id: null,
          file_count: 4,
          created_at: "2026-09-24T10:00:00Z",
          updated_at: "2026-09-24T10:00:00Z",
        },
      ],
    }),
  ),
];
