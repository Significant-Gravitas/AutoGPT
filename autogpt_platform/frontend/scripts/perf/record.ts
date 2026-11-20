/**
 * Re-records the copilot fixtures from a real local stack, for when the wire
 * format changes. Saves the session's messages as fixtures/session-200.json
 * and, with --message, one streamed reply as fixtures/reply-20-lines.sse.
 * Use a throwaway chat: the fixtures are committed.
 *
 *   npx tsx scripts/perf/record.ts --session-id <id> [--message "go"]
 */
import { randomUUID } from "crypto";
import fs from "fs";
import path from "path";
import { parseArgs } from "util";
import { request as playwrightRequest } from "@playwright/test";
import { signIn } from "./lib/auth";
import { FIXTURES_DIR } from "./lib/paths";

const MAX_SESSION_ROWS = 200;

function parseOptions() {
  const { values } = parseArgs({
    options: {
      "session-id": { type: "string" },
      message: { type: "string" },
      "base-url": { type: "string", default: "http://localhost:3000" },
      "backend-url": { type: "string", default: "http://localhost:8006" },
      "storage-state": { type: "string" },
      "session-out": { type: "string", default: "session-200.json" },
      "reply-out": { type: "string", default: "reply-20-lines.sse" },
    },
  });
  if (!values["session-id"]) throw new Error("Pass --session-id <id>.");
  return {
    sessionID: values["session-id"],
    message: values.message,
    baseURL: (values["base-url"] ?? "").replace(/\/$/, ""),
    backendURL: (values["backend-url"] ?? "").replace(/\/$/, ""),
    storageState: values["storage-state"],
    sessionOut: path.join(FIXTURES_DIR, values["session-out"] ?? ""),
    replyOut: path.join(FIXTURES_DIR, values["reply-out"] ?? ""),
  };
}

async function main() {
  const options = parseOptions();
  const request = await playwrightRequest.newContext({
    storageState: options.storageState,
  });
  try {
    await signIn(request, options.baseURL, Boolean(options.storageState));
    const session = await request.get(
      `${options.baseURL}/api/proxy/api/chat/sessions/${options.sessionID}?limit=${MAX_SESSION_ROWS}`,
    );
    if (!session.ok()) {
      throw new Error(`Session GET failed with ${session.status()}`);
    }
    const { messages } = (await session.json()) as { messages: unknown[] };
    fs.writeFileSync(
      options.sessionOut,
      `${JSON.stringify({ messages }, null, 2)}\n`,
    );
    console.log(`wrote ${messages.length} rows to ${options.sessionOut}`);
    if (!options.message) return;

    const token = await request.get(`${options.baseURL}/api/auth/token`);
    const { token: jwt } = (await token.json()) as { token?: string };
    if (!jwt) throw new Error("Could not mint a backend token");
    const stream = await request.post(
      `${options.backendURL}/api/chat/sessions/${options.sessionID}/stream`,
      {
        headers: { Authorization: `Bearer ${jwt}` },
        data: {
          message: options.message,
          is_user_message: true,
          context: null,
          file_ids: null,
          folder_ids: null,
          model: null,
          message_id: randomUUID(),
          voice: false,
          expert_kickoff: false,
        },
        timeout: 300_000,
      },
    );
    if (!stream.ok()) {
      throw new Error(`Stream POST failed with ${stream.status()}`);
    }
    fs.writeFileSync(options.replyOut, await stream.text());
    console.log(`wrote the streamed reply to ${options.replyOut}`);
  } finally {
    await request.dispose();
  }
}

main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.message : error);
  process.exit(1);
});
