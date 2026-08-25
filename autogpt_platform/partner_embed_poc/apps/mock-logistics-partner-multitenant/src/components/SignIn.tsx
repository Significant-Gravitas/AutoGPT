import { useState, type FormEvent } from "react";

import { initials } from "../helpers";
import type { DirectoryUser } from "../types";

interface Props {
  users: DirectoryUser[];
  accessRequired: boolean;
  busy: boolean;
  error: string | null;
  onUnlock: (code: string) => Promise<void>;
  onSignIn: (userID: string) => Promise<void>;
}

export function SignIn({
  users,
  accessRequired,
  busy,
  error,
  onUnlock,
  onSignIn,
}: Props) {
  const [accessCode, setAccessCode] = useState("");

  function handleUnlock(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void onUnlock(accessCode);
  }

  return (
    <main className="sign-in">
      <section className="sign-in-card">
        <div className="brand brand--login">
          <span className="brand-mark">RF</span>
          <div>
            <strong>Relay Freight OS</strong>
            <small>Secure partner workspace</small>
          </div>
        </div>
        <p className="eyebrow">Freight Operations Platform</p>
        <h1>
          {accessRequired ? "Open the Team Demo" : "Choose Your Demo Workspace"}
        </h1>
        <p>
          {accessRequired
            ? "Enter the shared team access code before choosing a synthetic freight workspace."
            : "Continue with an existing host identity. The embedded assistant never asks users to create a second account."}
        </p>
        {error && (
          <div className="error-banner" role="alert">
            {error}
          </div>
        )}
        {accessRequired ? (
          <form className="access-form" onSubmit={handleUnlock}>
            <label htmlFor="relay-demo-access">Team access code</label>
            <input
              id="relay-demo-access"
              name="demo-access-code"
              type="password"
              autoComplete="current-password"
              value={accessCode}
              aria-invalid={Boolean(error)}
              onChange={(event) => setAccessCode(event.target.value)}
            />
            <button type="submit" disabled={busy || !accessCode}>
              {busy ? "Checking access…" : "Continue to workspaces"}
            </button>
          </form>
        ) : (
          <div className="user-list">
            {users.map((user) => (
              <button
                key={user.id}
                type="button"
                disabled={busy}
                onClick={() => void onSignIn(user.id)}
              >
                <span>{initials(user.name)}</span>
                <div>
                  <strong>{user.name}</strong>
                  <small>{user.organizations.join(" · ")}</small>
                </div>
                <em>Continue</em>
              </button>
            ))}
          </div>
        )}
      </section>
    </main>
  );
}
