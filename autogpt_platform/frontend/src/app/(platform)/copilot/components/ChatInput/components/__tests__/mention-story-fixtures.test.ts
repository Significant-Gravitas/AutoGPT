import { listExpertCredentials } from "@/app/api/__generated__/endpoints/experts/experts";
import {
  getV1ListCredentials,
  getV1ListProviders,
} from "@/app/api/__generated__/endpoints/integrations/integrations";
import { listCopilotSkills } from "@/app/api/__generated__/endpoints/skills/skills";
import {
  listWorkspaceFiles,
  listWorkspaceFolders,
} from "@/app/api/__generated__/endpoints/workspace/workspace";
import { server } from "@/mocks/mock-server";
import { expect, it } from "vitest";
import { mentionStoryHandlers } from "../mentionStoryFixtures";

// No CI job runs the stories, so this is where their mocks meet the client's paths.
it.each([
  ["skills", () => listCopilotSkills(), "fix-issue"],
  ["credentials", () => getV1ListCredentials(), "Work Gmail"],
  ["providers", () => getV1ListProviders(), "Linear"],
  ["expert credentials", () => listExpertCredentials("expert-a"), "Work Gmail"],
  ["workspace files", () => listWorkspaceFiles(), "Team TODOs.md"],
  ["workspace folders", () => listWorkspaceFolders(), "Project files"],
])(
  "answers the mention story's %s on the path the client calls",
  async (_, read, expected) => {
    server.use(...mentionStoryHandlers);
    const response = await read();
    expect(response.status).toBe(200);
    expect(JSON.stringify(response.data)).toContain(expected);
  },
);
