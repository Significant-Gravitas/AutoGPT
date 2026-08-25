import { AutoGPTEmbeddedChat } from "@autogpt/embedded-chat";
import "@autogpt/embedded-chat/styles.css";
import { useEffect, useState } from "react";

import { SignIn } from "./components/SignIn";
import { Summary } from "./components/Summary";
import { SyncPanel } from "./components/SyncPanel";
import { initials } from "./helpers";
import type { DirectoryUser, Session, TokenResponse } from "./types";

const jobsByOrganization: Record<
  string,
  { reference: string; lane: string; status: string }[]
> = {
  "fd-account-77": [
    {
      reference: "NSF-24091",
      lane: "Shanghai → Liverpool",
      status: "Docs due",
    },
    {
      reference: "NSF-24102",
      lane: "Chicago → Manchester",
      status: "On track",
    },
    { reference: "NSF-24118", lane: "Rotterdam → Leeds", status: "Exception" },
  ],
  "fd-account-88": [
    {
      reference: "HRL-8824",
      lane: "Gothenburg → Felixstowe",
      status: "Customs",
    },
    { reference: "HRL-8831", lane: "Bilbao → Bristol", status: "On track" },
  ],
};

export default function App() {
  const [directory, setDirectory] = useState<DirectoryUser[]>([]);
  const [session, setSession] = useState<Session | null>(null);
  const [loading, setLoading] = useState(true);
  const [syncing, setSyncing] = useState(false);
  const [switching, setSwitching] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    void initialize();
  }, []);

  async function initialize() {
    const [directoryResponse, sessionResponse] = await Promise.all([
      fetch("/api/directory"),
      fetch("/api/session"),
    ]);
    if (directoryResponse.ok) {
      const body = (await directoryResponse.json()) as {
        users: DirectoryUser[];
      };
      setDirectory(body.users);
    }
    if (sessionResponse.ok) {
      setSession((await sessionResponse.json()) as Session);
    }
    setLoading(false);
  }

  async function signIn(userID: string) {
    setError(null);
    const response = await fetch("/api/session", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ userID }),
    });
    if (!response.ok) {
      setError("Unable to create the Forwarding Digital session.");
      return;
    }
    setSession((await response.json()) as Session);
  }

  async function signOut() {
    await fetch("/api/session", { method: "DELETE" });
    setSession(null);
  }

  async function switchOrganization(organizationID: string) {
    setSwitching(true);
    setError(null);
    const response = await fetch("/api/session/organization", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ organizationID }),
    });
    if (response.ok) setSession((await response.json()) as Session);
    else setError("You do not have access to that organization.");
    setSwitching(false);
  }

  async function refreshSession() {
    const response = await fetch("/api/session");
    if (response.ok) setSession((await response.json()) as Session);
  }

  async function syncTenant() {
    setSyncing(true);
    setError(null);
    const response = await fetch("/api/autogpt/sync", { method: "POST" });
    if (response.ok) await refreshSession();
    else setError("The AutoGPT tenant sync failed.");
    setSyncing(false);
  }

  async function getAccessToken() {
    const response = await fetch("/api/autogpt/token", { method: "POST" });
    if (!response.ok) throw new Error("Unable to authorize the assistant");
    const token = (await response.json()) as TokenResponse;
    await refreshSession();
    return token.access_token;
  }

  if (loading) {
    return <main className="loading">Loading Forwarding Digital…</main>;
  }
  if (!session) {
    return <SignIn users={directory} error={error} onSignIn={signIn} />;
  }

  const jobs = jobsByOrganization[session.activeOrganization.id] ?? [];

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-mark">FD</span>
          <div>
            <strong>forwarding digital</strong>
            <small>Multi-tenant partner demo</small>
          </div>
        </div>
        <nav aria-label="Primary navigation">
          <a className="active" href="#control-tower">
            Control tower
          </a>
          <a href="#shipments">Shipments</a>
          <a href="#documents">Documents</a>
          <a href="#reports">Reports</a>
          <a href="#automations">Automations</a>
        </nav>
        <div className="identity">
          <span>{initials(session.user.name)}</span>
          <div>
            <strong>{session.user.name}</strong>
            <small>{session.user.email}</small>
          </div>
        </div>
        <button className="text-button" onClick={signOut}>
          Sign out
        </button>
      </aside>

      <main className="workspace">
        <header className="topbar">
          <div>
            <p>Control tower</p>
            <h1>{session.activeOrganization.name}</h1>
          </div>
          <label className="tenant-picker">
            <span>Active organization</span>
            <select
              disabled={switching}
              value={session.activeOrganization.id}
              onChange={(event) => void switchOrganization(event.target.value)}
            >
              {session.organizations.map((organization) => (
                <option key={organization.id} value={organization.id}>
                  {organization.name}
                </option>
              ))}
            </select>
          </label>
        </header>

        {error && <div className="error-banner">{error}</div>}

        <section className="summary-grid" aria-label="Tenant summary">
          <Summary label="Open shipments" value={String(124 + jobs.length)} />
          <Summary
            label="Exceptions"
            value={jobs.length > 2 ? "7" : "2"}
            alert
          />
          <Summary label="Role" value={session.activeOrganization.role} />
          <Summary
            label="Assistant capabilities"
            value={String(session.activeOrganization.tools.length)}
          />
        </section>

        <section className="main-grid">
          <div className="operations">
            <div className="panel-heading">
              <div>
                <p>Live operations</p>
                <h2>Priority movements</h2>
              </div>
              <span className="tenant-id">{session.activeOrganization.id}</span>
            </div>
            <div className="job-list">
              {jobs.map((job) => (
                <article key={job.reference}>
                  <div>
                    <strong>{job.reference}</strong>
                    <span>{job.lane}</span>
                  </div>
                  <em>{job.status}</em>
                </article>
              ))}
            </div>
          </div>

          <SyncPanel
            mapping={session.sync}
            syncing={syncing}
            onSync={syncTenant}
          />

          <div className="chat-panel">
            <div className="tenant-lock">
              Assistant locked to{" "}
              <strong>{session.activeOrganization.name}</strong>
            </div>
            <AutoGPTEmbeddedChat
              key={session.user.id + ":" + session.activeOrganization.id}
              apiBaseURL=""
              brandName="Forwarding Digital"
              getAccessToken={getAccessToken}
              title="Forwarding Assistant"
            />
          </div>
        </section>
      </main>
    </div>
  );
}
