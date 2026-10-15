import {
  createContext,
  use,
  useEffect,
  useEffectEvent,
  useRef,
  useState,
  type ReactNode,
} from "react";

import { BackendClient } from "../../api/backend";
import {
  HttpError,
  SessionEndedError,
  SessionRejectedError,
} from "../../api/errors";
import { stableDigest } from "../../lib/ids";
import { serverPartitionKey, type ServerConfig } from "../servers/serverConfig";
import {
  findServer,
  forgetCustomServer,
  loadCustomServers,
  loadSelection,
  saveCustomServer,
  saveSelection,
} from "../servers/serverStore";
import { SessionAccessTokens } from "./accessTokens";
import {
  exchangeAccessToken,
  revokeSession,
  signInWithEmail,
  SignInError,
} from "./authApi";
import {
  credentialKey,
  deleteCredential,
  readCredential,
  wipeCredentialsLeftByPreviousInstall,
  writeCredential,
  type StoredCredential,
} from "./credentialStore";

export interface Account {
  /** Keychain key of this account's credential; also partitions in-memory caches. */
  key: string;
  userID: string;
  email: string;
  name: string | null;
}

export type AuthState =
  | { status: "restoring" }
  | { status: "signedOut"; server: ServerConfig | null; notice: string | null }
  | {
      status: "signedIn";
      server: ServerConfig;
      account: Account;
      api: BackendClient;
    };

interface SessionContextValue {
  state: AuthState;
  customServers: ServerConfig[];
  chooseServer: (server: ServerConfig) => Promise<void>;
  rememberServer: (server: ServerConfig) => Promise<void>;
  forgetServer: (serverID: string) => Promise<void>;
  signIn: (email: string, password: string) => Promise<void>;
  signOut: () => Promise<void>;
}

const SessionContext = createContext<SessionContextValue | null>(null);

export function SessionProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<AuthState>({ status: "restoring" });
  const [customServers, setCustomServers] = useState<ServerConfig[]>([]);
  const live = useRef<{
    tokens: SessionAccessTokens;
    credential: StoredCredential;
  } | null>(null);
  const stateRef = useRef(state);

  useEffect(() => {
    stateRef.current = state;
  }, [state]);

  function activate(
    server: ServerConfig,
    account: Account,
    credential: StoredCredential,
  ) {
    live.current?.tokens.dispose();
    const tokens = new SessionAccessTokens(server, credential.sessionToken, {
      onSessionEnded: () =>
        void endLocally(server, account, "Your session ended. Sign in again."),
      onSessionRenewed: (sessionToken) => void renew(account, sessionToken),
    });
    live.current = { tokens, credential };
    setState({
      status: "signedIn",
      server,
      account,
      api: new BackendClient(server, tokens),
    });
    return tokens;
  }

  async function endLocally(
    server: ServerConfig,
    account: Account,
    notice: string | null,
  ) {
    const current = live.current;
    if (!current || current.credential.userID !== account.userID) return;
    live.current = null;
    current.tokens.dispose();
    // The local credential goes first, so nothing can reuse it whatever happens next.
    await deleteCredential(account.key).catch(() => {});
    await saveSelection({ serverID: server.id, accountKey: null }).catch(
      () => {},
    );
    setState({ status: "signedOut", server, notice });
    return current.credential.sessionToken;
  }

  async function renew(account: Account, sessionToken: string) {
    const current = live.current;
    if (!current || current.credential.userID !== account.userID) return;
    const credential = {
      ...current.credential,
      sessionToken,
      savedAt: new Date().toISOString(),
    };
    live.current = { ...current, credential };
    await writeCredential(account.key, credential).catch(() => {});
  }

  const applyRestored = useEffectEvent((restored: Restored | null) => {
    if (!restored) {
      setState({
        status: "signedOut",
        server: null,
        notice: "Couldn't read your saved sign-in.",
      });
      return;
    }
    setCustomServers(restored.customServers);
    if (restored.signedIn) {
      const { server, account, credential } = restored.signedIn;
      activate(server, account, credential);
    } else {
      setState({ status: "signedOut", server: restored.server, notice: null });
    }
  });

  useEffect(() => {
    let cancelled = false;
    restore()
      .then((restored) => {
        if (!cancelled) applyRestored(restored);
      })
      .catch(() => {
        if (!cancelled) applyRestored(null);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  async function chooseServer(server: ServerConfig) {
    if (stateRef.current.status === "signedIn") return;
    await saveSelection({ serverID: server.id, accountKey: null });
    setState({ status: "signedOut", server, notice: null });
  }

  async function rememberServer(server: ServerConfig) {
    setCustomServers(await saveCustomServer(server));
    await chooseServer(server);
  }

  async function forgetServer(serverID: string) {
    setCustomServers(await forgetCustomServer(serverID));
    const current = stateRef.current;
    if (current.status === "signedOut" && current.server?.id === serverID) {
      setState({ status: "signedOut", server: null, notice: null });
    }
  }

  async function signIn(email: string, password: string) {
    const current = stateRef.current;
    if (current.status !== "signedOut" || !current.server) {
      throw new SignInError("rejected", "Choose a server first.");
    }
    const server = current.server;
    const signedIn = await signInWithEmail(server, email.trim(), password);
    const access = await verifyNativeSession(server, signedIn.sessionToken);
    const accountKey = await stableDigest(signedIn.userID);
    const key = credentialKey(await serverPartitionKey(server), accountKey);
    const credential: StoredCredential = {
      version: 1,
      sessionToken: signedIn.sessionToken,
      authOrigin: server.authUrl,
      apiOrigin: server.apiUrl,
      userID: signedIn.userID,
      email: signedIn.email,
      name: signedIn.name,
      savedAt: new Date().toISOString(),
    };
    try {
      await writeCredential(key, credential);
      await saveSelection({ serverID: server.id, accountKey });
    } catch {
      void revokeSession(server, signedIn.sessionToken);
      throw new SignInError(
        "unavailable",
        "Couldn't save your sign-in on this device. Try again.",
      );
    }
    const account = {
      key,
      userID: signedIn.userID,
      email: signedIn.email,
      name: signedIn.name,
    };
    activate(server, account, credential).seed(access.token, access.expiresAt);
  }

  async function signOut() {
    const current = stateRef.current;
    if (current.status !== "signedIn") return;
    const sessionToken = await endLocally(
      current.server,
      current.account,
      null,
    );
    if (sessionToken) void revokeSession(current.server, sessionToken);
  }

  return (
    <SessionContext
      value={{
        state,
        customServers,
        chooseServer,
        rememberServer,
        forgetServer,
        signIn,
        signOut,
      }}
    >
      {children}
    </SessionContext>
  );
}

export function useSession(): SessionContextValue {
  const value = use(SessionContext);
  if (!value) throw new Error("useSession must be used inside SessionProvider");
  return value;
}

/** For screens that only exist while signed in. */
export function useSignedIn() {
  const { state } = useSession();
  if (state.status !== "signedIn")
    throw new Error("This screen requires a signed-in account");
  return state;
}

interface Restored {
  customServers: ServerConfig[];
  server: ServerConfig | null;
  signedIn: {
    server: ServerConfig;
    account: Account;
    credential: StoredCredential;
  } | null;
}

async function restore(): Promise<Restored> {
  await wipeCredentialsLeftByPreviousInstall();
  const customServers = await loadCustomServers();
  const selection = await loadSelection();
  const server = selection ? await findServer(selection.serverID) : null;
  if (!server || !selection?.accountKey) {
    return { customServers, server, signedIn: null };
  }
  const key = credentialKey(
    await serverPartitionKey(server),
    selection.accountKey,
  );
  const credential = await readCredential(key);
  // A credential is only used with the exact origins it was issued for.
  if (
    !credential ||
    credential.authOrigin !== server.authUrl ||
    credential.apiOrigin !== server.apiUrl
  ) {
    await saveSelection({ serverID: server.id, accountKey: null });
    return { customServers, server, signedIn: null };
  }
  const account = {
    key,
    userID: credential.userID,
    email: credential.email,
    name: credential.name,
  };
  return { customServers, server, signedIn: { server, account, credential } };
}

/**
 * Proves the new session works end to end before it is stored: the auth
 * server must issue an access token for it, and the API must accept that
 * token. A transient API failure doesn't block sign-in; a refusal does.
 */
async function verifyNativeSession(server: ServerConfig, sessionToken: string) {
  let access;
  try {
    access = await exchangeAccessToken(server, sessionToken);
  } catch (error) {
    void revokeSession(server, sessionToken);
    if (error instanceof SessionEndedError) {
      throw new SignInError(
        "incompatible",
        "This server signed you in but didn't accept the app's session.",
      );
    }
    throw new SignInError(
      "unavailable",
      "The server couldn't finish signing you in. Try again.",
    );
  }
  const probe = new BackendClient(server, {
    getAccessToken: async () => access.token,
  });
  try {
    await probe.json("/api/chat/sessions", { query: { limit: 1 } });
  } catch (error) {
    const refused =
      error instanceof SessionRejectedError ||
      (error instanceof HttpError &&
        (error.status === 401 || error.status === 403));
    if (refused) {
      void revokeSession(server, sessionToken);
      throw new SignInError(
        "incompatible",
        "The API address didn't accept this sign-in. Check that both addresses belong to the same AutoGPT server.",
      );
    }
  }
  return access;
}
