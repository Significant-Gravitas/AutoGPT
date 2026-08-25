import { AutoGPTEmbeddedChat } from "@autogpt/embedded-chat";
import "@autogpt/embedded-chat/styles.css";
import { useEffect, useState } from "react";

interface SessionUser {
  name: string;
  email: string;
  accountName: string;
  roles: string[];
}

interface TokenResponse {
  access_token: string;
}

const jobs = [
  {
    ref: "NSF-24091",
    mode: "Sea",
    eta: "26 Aug",
    route: "CNSHA → GBLIV",
    status: "Docs due",
  },
  {
    ref: "NSF-24102",
    mode: "Air",
    eta: "25 Aug",
    route: "USORD → GBMAN",
    status: "On track",
  },
  {
    ref: "NSF-24118",
    mode: "Road",
    eta: "24 Aug",
    route: "NLRTM → GBLBA",
    status: "Exception",
  },
];

export default function App() {
  const [user, setUser] = useState<SessionUser | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    void loadSession();
  }, []);

  async function loadSession() {
    const response = await fetch("/api/session");
    if (response.ok) setUser((await response.json()) as SessionUser);
    setLoading(false);
  }

  async function signIn() {
    const response = await fetch("/api/session", { method: "POST" });
    if (response.ok) setUser((await response.json()) as SessionUser);
  }

  async function signOut() {
    await fetch("/api/session", { method: "DELETE" });
    setUser(null);
  }

  async function getAccessToken() {
    const response = await fetch("/api/autogpt/token", { method: "POST" });
    if (!response.ok) throw new Error("Unable to authorize the assistant");
    const token = (await response.json()) as TokenResponse;
    return token.access_token;
  }

  if (loading)
    return <main className="loading">Loading Example Logistics…</main>;
  if (!user) return <SignIn onSignIn={signIn} />;

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <span>FD</span>
          <strong>example logistics</strong>
        </div>
        <nav aria-label="Primary navigation">
          <a className="active" href="#dashboard">
            Dashboard
          </a>
          <a href="#jobs">Jobs</a>
          <a href="#documents">Documents</a>
          <a href="#reports">Reports</a>
        </nav>
        <button className="sign-out" onClick={signOut}>
          Sign out
        </button>
      </aside>
      <main className="workspace">
        <header className="topbar">
          <div>
            <p>Operations</p>
            <h1>Good morning, {user.name.split(" ")[0]}</h1>
          </div>
          <div className="account">
            <span>{user.accountName}</span>
            <small>{user.email}</small>
          </div>
        </header>
        <section className="metrics" aria-label="Operational summary">
          <Metric
            label="Active jobs"
            value="148"
            detail="12 arriving this week"
          />
          <Metric
            label="Exceptions"
            value="7"
            detail="3 need attention"
            alert
          />
          <Metric
            label="Documents due"
            value="23"
            detail="Before close of play"
          />
        </section>
        <section className="content-grid">
          <div className="jobs-card">
            <div className="section-heading">
              <div>
                <p>Live operations</p>
                <h2>Upcoming arrivals</h2>
              </div>
              <button>View all jobs</button>
            </div>
            <table>
              <thead>
                <tr>
                  <th>Reference</th>
                  <th>Mode</th>
                  <th>Route</th>
                  <th>ETA</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {jobs.map((job) => (
                  <tr key={job.ref}>
                    <td>
                      <strong>{job.ref}</strong>
                    </td>
                    <td>{job.mode}</td>
                    <td>{job.route}</td>
                    <td>{job.eta}</td>
                    <td>
                      <span
                        className={`job-status job-status--${job.status.toLowerCase().replace(" ", "-")}`}
                      >
                        {job.status}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <AutoGPTEmbeddedChat
            apiBaseURL=""
            brandName="Example Logistics"
            getAccessToken={getAccessToken}
            title="Forwarding Assistant"
          />
        </section>
      </main>
    </div>
  );
}

function SignIn({ onSignIn }: { onSignIn: () => Promise<void> }) {
  return (
    <main className="sign-in">
      <div className="sign-in__card">
        <div className="brand brand--dark">
          <span>FD</span>
          <strong>example logistics</strong>
        </div>
        <p>Partner embed proof of concept</p>
        <h1>Northstar Freight workspace</h1>
        <p>
          Sign in to the mock Example Logistics account. No AutoGPT login is
          used.
        </p>
        <button onClick={() => void onSignIn()}>Continue as Alex Morgan</button>
      </div>
    </main>
  );
}

function Metric({
  label,
  value,
  detail,
  alert = false,
}: {
  label: string;
  value: string;
  detail: string;
  alert?: boolean;
}) {
  return (
    <article className={alert ? "metric metric--alert" : "metric"}>
      <p>{label}</p>
      <strong>{value}</strong>
      <span>{detail}</span>
    </article>
  );
}
