import { asString } from "../../lib/json";
import { readJSON, writeJSON } from "../../platform/kv";
import {
  CLOUD_SERVER,
  isServerConfig,
  type ServerConfig,
} from "./serverConfig";

const SERVERS_KEY = "servers.v1";
const SELECTION_KEY = "selection.v1";

/** Which server is chosen, and which of its accounts (by partition key) is signed in. */
export interface Selection {
  serverID: string;
  accountKey: string | null;
}

function isServerList(value: unknown): value is ServerConfig[] {
  return Array.isArray(value) && value.every(isServerConfig);
}

function isSelection(value: unknown): value is Selection {
  if (!value || typeof value !== "object") return false;
  const v = value as Record<string, unknown>;
  return (
    typeof v.serverID === "string" &&
    (v.accountKey === null || asString(v.accountKey) !== null)
  );
}

export async function loadCustomServers(): Promise<ServerConfig[]> {
  const servers = (await readJSON(SERVERS_KEY, isServerList)) ?? [];
  return servers.filter((server) => server.kind === "custom");
}

export async function saveCustomServer(
  server: ServerConfig,
): Promise<ServerConfig[]> {
  const others = (await loadCustomServers()).filter((s) => s.id !== server.id);
  const next = [server, ...others].slice(0, 10);
  await writeJSON(SERVERS_KEY, next);
  return next;
}

export async function forgetCustomServer(
  serverID: string,
): Promise<ServerConfig[]> {
  const next = (await loadCustomServers()).filter((s) => s.id !== serverID);
  await writeJSON(SERVERS_KEY, next);
  return next;
}

export async function findServer(
  serverID: string,
): Promise<ServerConfig | null> {
  if (serverID === CLOUD_SERVER.id) return CLOUD_SERVER;
  return (await loadCustomServers()).find((s) => s.id === serverID) ?? null;
}

export async function loadSelection(): Promise<Selection | null> {
  return readJSON(SELECTION_KEY, isSelection);
}

export async function saveSelection(selection: Selection): Promise<void> {
  await writeJSON(SELECTION_KEY, selection);
}
