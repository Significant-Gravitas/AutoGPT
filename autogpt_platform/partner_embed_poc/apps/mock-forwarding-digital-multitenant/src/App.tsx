import {
  AutoGPTEmbeddedChat,
  type AutoGPTEmbeddedChatTheme,
} from "@autogpt/embedded-chat";
import "@autogpt/embedded-chat/styles.css";
import { useEffect, useState } from "react";

import { SignIn } from "./components/SignIn";
import { Summary } from "./components/Summary";
import { SyncPanel } from "./components/SyncPanel";
import { assistantNoticeFor, documentsForJobs, initials } from "./helpers";
import type { DirectoryUser, Session, TokenResponse } from "./types";

type PageID = "overview" | "shipments" | "documents" | "automations";

interface Job {
  reference: string;
  lane: string;
  customer: string;
  eta: string;
  mode: string;
  status: string;
}

const navigation: { id: PageID; label: string }[] = [
  { id: "overview", label: "Overview" },
  { id: "shipments", label: "Shipments" },
  { id: "documents", label: "Documents" },
  { id: "automations", label: "Automations" },
];

const jobsByOrganization: Record<string, Job[]> = {
  "fd-account-77": [
    {
      reference: "NSF-24091",
      lane: "Shanghai → Liverpool",
      customer: "Hawthorn Retail",
      eta: "2026-08-26T09:30:00Z",
      mode: "Ocean",
      status: "Docs due",
    },
    {
      reference: "NSF-24102",
      lane: "Chicago → Manchester",
      customer: "Mason Industrial",
      eta: "2026-08-27T15:15:00Z",
      mode: "Air",
      status: "On track",
    },
    {
      reference: "NSF-24118",
      lane: "Rotterdam → Leeds",
      customer: "Calder Foods",
      eta: "2026-08-28T06:45:00Z",
      mode: "Road",
      status: "Exception",
    },
  ],
  "fd-account-88": [
    {
      reference: "HRL-8824",
      lane: "Gothenburg → Felixstowe",
      customer: "Seabrook Components",
      eta: "2026-08-26T12:20:00Z",
      mode: "Ocean",
      status: "Customs",
    },
    {
      reference: "HRL-8831",
      lane: "Bilbao → Bristol",
      customer: "Redcliffe Home",
      eta: "2026-08-29T10:00:00Z",
      mode: "Road",
      status: "On track",
    },
  ],
};

const suggestionsByPage: Record<PageID, string[]> = {
  overview: [
    "Summarize today's exceptions and name the jobs that need attention.",
    "List the next 2 arrivals with ETA and current exception status.",
  ],
  shipments: [
    "Compare the active shipment lanes and flag the highest operational risk.",
    "List the next 2 arrivals for this tenant. Do not invent data.",
  ],
  documents: [
    "Create an arrival-notice checklist for the next eligible ocean shipment.",
    "Find jobs with missing documents and produce an exception summary.",
  ],
  automations: [
    "Create and save a calculator agent that adds 10 to one numeric input using only enabled blocks.",
    "List my saved agents and schedules, including the next run time.",
    "Schedule the saved calculator agent for every Monday at 09:00 UTC.",
  ],
};

const assistantTheme: AutoGPTEmbeddedChatTheme = {
  background: "#f5f7fb",
  foreground: "#15233c",
  surface: "#ffffff",
  surfaceMuted: "#edf2fb",
  accent: "#2458d3",
  accentForeground: "#ffffff",
  border: "#dce4f2",
  danger: "#b42318",
  radius: "16px",
  shadow: "0 24px 70px rgb(20 46 92 / 18%)",
};

export default function App() {
  const [directory, setDirectory] = useState<DirectoryUser[]>([]);
  const [session, setSession] = useState<Session | null>(null);
  const [activePage, setActivePage] = useState<PageID>(() =>
    pageFromHash(window.location.hash),
  );
  const [assistantOpen, setAssistantOpen] = useState(false);
  const [assistantNotice, setAssistantNotice] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [accessRequired, setAccessRequired] = useState(false);
  const [accessBusy, setAccessBusy] = useState(false);
  const [syncing, setSyncing] = useState(false);
  const [switching, setSwitching] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    void initialize();
  }, []);

  useEffect(() => {
    function handleHashChange() {
      setActivePage(pageFromHash(window.location.hash));
    }

    window.addEventListener("hashchange", handleHashChange);
    return () => window.removeEventListener("hashchange", handleHashChange);
  }, []);

  async function initialize() {
    try {
      const accessResponse = await fetch("/api/demo-access");
      if (!accessResponse.ok) throw new Error("Demo access check failed");
      const access = (await accessResponse.json()) as {
        required: boolean;
        authorized: boolean;
      };
      if (access.required && !access.authorized) {
        setAccessRequired(true);
        return;
      }
      await loadWorkspace();
    } catch {
      setError("Relay Freight OS could not load the team demo.");
    } finally {
      setLoading(false);
    }
  }

  async function loadWorkspace() {
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
  }

  async function unlockDemo(code: string) {
    setAccessBusy(true);
    setError(null);
    try {
      const response = await fetch("/api/demo-access", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ code }),
      });
      if (!response.ok) {
        setError("That team access code was not accepted.");
        return;
      }
      setAccessRequired(false);
      await loadWorkspace();
      window.requestAnimationFrame(() => window.scrollTo(0, 0));
    } catch {
      setError("Relay Freight OS could not verify demo access.");
    } finally {
      setAccessBusy(false);
    }
  }

  async function signIn(userID: string) {
    setError(null);
    const response = await fetch("/api/session", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ userID }),
    });
    if (!response.ok) {
      setError("Relay Freight OS could not start this workspace session.");
      return;
    }
    setSession((await response.json()) as Session);
  }

  async function signOut() {
    await fetch("/api/session", { method: "DELETE" });
    setSession(null);
    setAssistantOpen(false);
  }

  async function switchOrganization(organizationID: string) {
    setSwitching(true);
    setError(null);
    const response = await fetch("/api/session/organization", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ organizationID }),
    });
    if (response.ok) {
      setSession((await response.json()) as Session);
      setAssistantNotice(null);
    } else {
      setError("You do not have access to that organization.");
    }
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
    else setError("The secure AI workspace sync failed. Try again.");
    setSyncing(false);
  }

  async function getAccessToken() {
    const response = await fetch("/api/autogpt/token", { method: "POST" });
    if (!response.ok) throw new Error("Unable to authorize the assistant");
    const token = (await response.json()) as TokenResponse;
    await refreshSession();
    return token.access_token;
  }

  function navigate(page: PageID) {
    window.location.hash = page;
    setActivePage(page);
    setAssistantNotice(null);
  }

  function openAssistant() {
    setAssistantOpen(true);
  }

  function handleAssistantNavigation(href: string) {
    navigate("automations");
    setAssistantNotice(assistantNoticeFor(href));
    setAssistantOpen(false);
  }

  if (loading) {
    return (
      <main className="loading" aria-live="polite">
        Loading Relay Freight OS…
      </main>
    );
  }
  if (!session) {
    return (
      <SignIn
        users={directory}
        accessRequired={accessRequired}
        busy={accessBusy}
        error={error}
        onUnlock={unlockDemo}
        onSignIn={signIn}
      />
    );
  }

  const jobs = jobsByOrganization[session.activeOrganization.id] ?? [];
  const canManageAgents =
    session.activeOrganization.tools.includes("agents.create") &&
    session.activeOrganization.tools.includes("agents.schedule");

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true">
            RF
          </span>
          <div>
            <strong>Relay Freight OS</strong>
            <small>Operations workspace</small>
          </div>
        </div>
        <nav aria-label="Primary navigation">
          {navigation.map((item) => (
            <a
              key={item.id}
              href={`#${item.id}`}
              aria-current={activePage === item.id ? "page" : undefined}
              onClick={() => navigate(item.id)}
            >
              {item.label}
            </a>
          ))}
        </nav>
        <div className="sidebar-footer">
          <div className="identity">
            <span>{initials(session.user.name)}</span>
            <div>
              <strong>{session.user.name}</strong>
              <small>{session.user.email}</small>
            </div>
          </div>
          <button type="button" className="text-button" onClick={signOut}>
            Sign Out
          </button>
        </div>
      </aside>

      <main className="workspace" id="main-content">
        <header className="topbar">
          <div>
            <p>{pageEyebrow(activePage)}</p>
            <h1>{pageTitle(activePage)}</h1>
          </div>
          <div className="topbar-actions">
            <label className="tenant-picker">
              <span>Active Organization</span>
              <select
                name="active-organization"
                disabled={switching}
                value={session.activeOrganization.id}
                onChange={(event) =>
                  void switchOrganization(event.target.value)
                }
              >
                {session.organizations.map((organization) => (
                  <option key={organization.id} value={organization.id}>
                    {organization.name}
                  </option>
                ))}
              </select>
            </label>
            {activePage !== "automations" ? (
              <button
                type="button"
                className="assistant-button"
                onClick={openAssistant}
              >
                Open Operations Copilot
              </button>
            ) : null}
          </div>
        </header>

        {error ? (
          <div className="error-banner" role="alert">
            {error}
          </div>
        ) : null}
        {assistantNotice ? (
          <div className="success-banner" role="status">
            {assistantNotice}
          </div>
        ) : null}

        <WorkspacePage
          activePage={activePage}
          canManageAgents={canManageAgents}
          jobs={jobs}
          session={session}
          syncing={syncing}
          onNavigate={navigate}
          onOpenAssistant={openAssistant}
          onSync={syncTenant}
          getAccessToken={getAccessToken}
          onAssistantNavigation={handleAssistantNavigation}
        />
      </main>

      {activePage !== "automations" ? (
        <button
          type="button"
          className="assistant-launcher"
          aria-label="Open Operations Copilot"
          onClick={openAssistant}
        >
          <span aria-hidden="true">✦</span>
          Ask Copilot
        </button>
      ) : null}

      {assistantOpen && activePage !== "automations" ? (
        <div className="assistant-overlay">
          <button
            type="button"
            className="assistant-backdrop"
            aria-label="Close Operations Copilot"
            onClick={() => setAssistantOpen(false)}
          />
          <aside
            className="assistant-drawer"
            role="dialog"
            aria-modal="true"
            aria-label="Operations Copilot"
          >
            <div className="drawer-heading">
              <div>
                <p>Contextual Assistant</p>
                <strong>{session.activeOrganization.name}</strong>
              </div>
              <button
                type="button"
                aria-label="Close Operations Copilot"
                onClick={() => setAssistantOpen(false)}
              >
                ×
              </button>
            </div>
            <Assistant
              getAccessToken={getAccessToken}
              onNavigate={handleAssistantNavigation}
              prompts={suggestionsByPage[activePage]}
              session={session}
            />
          </aside>
        </div>
      ) : null}
    </div>
  );
}

interface WorkspacePageProps {
  activePage: PageID;
  canManageAgents: boolean;
  jobs: Job[];
  session: Session;
  syncing: boolean;
  onNavigate: (page: PageID) => void;
  onOpenAssistant: () => void;
  onSync: () => Promise<void>;
  getAccessToken: () => Promise<string>;
  onAssistantNavigation: (href: string) => void;
}

function WorkspacePage({
  activePage,
  canManageAgents,
  jobs,
  session,
  syncing,
  onNavigate,
  onOpenAssistant,
  onSync,
  getAccessToken,
  onAssistantNavigation,
}: WorkspacePageProps) {
  if (activePage === "shipments") {
    return <ShipmentsPage jobs={jobs} onOpenAssistant={onOpenAssistant} />;
  }
  if (activePage === "documents") {
    return <DocumentsPage jobs={jobs} onOpenAssistant={onOpenAssistant} />;
  }
  if (activePage === "automations") {
    return (
      <AutomationsPage
        canManageAgents={canManageAgents}
        getAccessToken={getAccessToken}
        onAssistantNavigation={onAssistantNavigation}
        onSync={onSync}
        session={session}
        syncing={syncing}
      />
    );
  }
  return (
    <OverviewPage
      jobs={jobs}
      session={session}
      onNavigate={onNavigate}
      onOpenAssistant={onOpenAssistant}
    />
  );
}

interface OverviewPageProps {
  jobs: Job[];
  session: Session;
  onNavigate: (page: PageID) => void;
  onOpenAssistant: () => void;
}

function OverviewPage({
  jobs,
  session,
  onNavigate,
  onOpenAssistant,
}: OverviewPageProps) {
  return (
    <>
      <section className="summary-grid" aria-label="Organization Summary">
        <Summary label="Open Shipments" value={String(124 + jobs.length)} />
        <Summary label="Exceptions" value={jobs.length > 2 ? "7" : "2"} alert />
        <Summary
          label="Arriving This Week"
          value={String(Math.max(2, jobs.length))}
        />
        <Summary
          label="Automation Coverage"
          value={session.activeOrganization.role === "manager" ? "68%" : "View"}
        />
      </section>

      <section className="overview-grid">
        <section className="panel movements-panel">
          <div className="panel-heading">
            <div>
              <p>Live Operations</p>
              <h2>Priority Movements</h2>
            </div>
            <button type="button" onClick={() => onNavigate("shipments")}>
              View All Shipments
            </button>
          </div>
          <div className="job-list">
            {jobs.map((job) => (
              <article key={job.reference}>
                <div>
                  <strong>{job.reference}</strong>
                  <span>{job.lane}</span>
                </div>
                <time dateTime={job.eta}>{formatETA(job.eta)}</time>
                <em data-status={statusKind(job.status)}>{job.status}</em>
              </article>
            ))}
          </div>
        </section>

        <aside className="copilot-callout">
          <span className="copilot-orbit" aria-hidden="true">
            ✦
          </span>
          <p>Operations Copilot</p>
          <h2>Turn live freight data into action.</h2>
          <p>
            Review tenant-scoped MCP data, create agents from enabled blocks,
            run them now, or schedule recurring work.
          </p>
          <button type="button" onClick={onOpenAssistant}>
            Ask About Today
          </button>
          <button
            type="button"
            className="secondary-action"
            onClick={() => onNavigate("automations")}
          >
            Build an Automation
          </button>
        </aside>
      </section>
    </>
  );
}

interface ShipmentsPageProps {
  jobs: Job[];
  onOpenAssistant: () => void;
}

function ShipmentsPage({ jobs, onOpenAssistant }: ShipmentsPageProps) {
  return (
    <section className="panel table-panel">
      <div className="panel-heading">
        <div>
          <p>Active Book</p>
          <h2>Shipment Register</h2>
        </div>
        <button type="button" onClick={onOpenAssistant}>
          Analyze With Copilot
        </button>
      </div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th scope="col">Reference</th>
              <th scope="col">Customer</th>
              <th scope="col">Lane</th>
              <th scope="col">Mode</th>
              <th scope="col">ETA</th>
              <th scope="col">Status</th>
            </tr>
          </thead>
          <tbody>
            {jobs.map((job) => (
              <tr key={job.reference}>
                <td>
                  <strong>{job.reference}</strong>
                </td>
                <td>{job.customer}</td>
                <td>{job.lane}</td>
                <td>{job.mode}</td>
                <td>
                  <time dateTime={job.eta}>{formatETA(job.eta)}</time>
                </td>
                <td>
                  <span
                    className="status-pill"
                    data-status={statusKind(job.status)}
                  >
                    {job.status}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

interface DocumentsPageProps {
  jobs: Job[];
  onOpenAssistant: () => void;
}

function DocumentsPage({ jobs, onOpenAssistant }: DocumentsPageProps) {
  const documents = documentsForJobs(jobs);

  return (
    <section className="documents-layout">
      <div className="panel">
        <div className="panel-heading">
          <div>
            <p>Document Desk</p>
            <h2>Upcoming Work</h2>
          </div>
          <button type="button" onClick={onOpenAssistant}>
            Generate With Copilot
          </button>
        </div>
        <div className="document-list">
          {documents.map((document) => (
            <article key={document.name}>
              <span className="document-icon" aria-hidden="true">
                DOC
              </span>
              <div>
                <strong>{document.name}</strong>
                <span>{document.type}</span>
              </div>
              <em>{document.state}</em>
            </article>
          ))}
        </div>
      </div>
      <aside className="context-panel">
        <p>Embedded Workflow</p>
        <h2>Create documents from the same tenant-scoped conversation.</h2>
        <p>
          Generated files remain attached to the chat session and download
          through the host-authorized artifact surface.
        </p>
        <button type="button" onClick={onOpenAssistant}>
          Open Document Assistant
        </button>
      </aside>
    </section>
  );
}

interface AutomationsPageProps {
  canManageAgents: boolean;
  getAccessToken: () => Promise<string>;
  onAssistantNavigation: (href: string) => void;
  onSync: () => Promise<void>;
  session: Session;
  syncing: boolean;
}

function AutomationsPage({
  canManageAgents,
  getAccessToken,
  onAssistantNavigation,
  onSync,
  session,
  syncing,
}: AutomationsPageProps) {
  return (
    <section className="automation-layout">
      <aside className="automation-guide">
        <p>Agent Workspace</p>
        <h2>Build, run, and schedule in conversation.</h2>
        <p>
          The assistant can only use blocks and tools granted to this user by
          the host application.
        </p>
        <ol>
          <li>
            <strong>Describe the outcome</strong>
            <span>Use a suggested prompt or write your own workflow.</span>
          </li>
          <li>
            <strong>Review tool activity</strong>
            <span>
              Inspect block discovery, validation, runs, and schedules.
            </span>
          </li>
          <li>
            <strong>Open the saved resource</strong>
            <span>
              Host-owned navigation keeps users inside this application.
            </span>
          </li>
        </ol>
        {!canManageAgents ? (
          <p className="permission-note" role="status">
            Your current role can analyze operations but cannot create, run, or
            schedule agents.
          </p>
        ) : (
          <p className="permission-note success" role="status">
            Manager controls enabled: create, run, and schedule.
          </p>
        )}
        <SyncPanel mapping={session.sync} syncing={syncing} onSync={onSync} />
      </aside>
      <div className="inline-assistant">
        <div className="tenant-lock">
          <span aria-hidden="true">●</span>
          Scoped to <strong>{session.activeOrganization.name}</strong>
          <small>{session.activeOrganization.tools.length} capabilities</small>
        </div>
        <Assistant
          getAccessToken={getAccessToken}
          onNavigate={onAssistantNavigation}
          prompts={suggestionsByPage.automations}
          session={session}
        />
      </div>
    </section>
  );
}

interface AssistantProps {
  getAccessToken: () => Promise<string>;
  onNavigate: (href: string) => void;
  prompts: string[];
  session: Session;
}

function Assistant({
  getAccessToken,
  onNavigate,
  prompts,
  session,
}: AssistantProps) {
  return (
    <AutoGPTEmbeddedChat
      key={session.user.id + ":" + session.activeOrganization.id}
      apiBaseURL=""
      brandName="Relay Freight AI"
      getAccessToken={getAccessToken}
      onNavigate={onNavigate}
      suggestedPrompts={prompts}
      theme={assistantTheme}
      title="Operations Copilot"
    />
  );
}

function pageFromHash(hash: string): PageID {
  const candidate = hash.replace("#", "") as PageID;
  return navigation.some((item) => item.id === candidate)
    ? candidate
    : "overview";
}

function pageEyebrow(page: PageID): string {
  if (page === "shipments") return "Shipment Operations";
  if (page === "documents") return "Document Control";
  if (page === "automations") return "AI & Automation";
  return "Live Operations";
}

function pageTitle(page: PageID): string {
  if (page === "shipments") return "Shipments";
  if (page === "documents") return "Documents";
  if (page === "automations") return "Automations";
  return "Control Tower";
}

function formatETA(value: string): string {
  return new Intl.DateTimeFormat("en-GB", {
    day: "numeric",
    month: "short",
    timeZone: "UTC",
  }).format(new Date(value));
}

function statusKind(status: string): "good" | "warning" | "danger" {
  if (/exception|customs|missing/i.test(status)) return "danger";
  if (/due|pending|review/i.test(status)) return "warning";
  return "good";
}
