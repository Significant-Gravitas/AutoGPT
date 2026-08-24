import { initials } from "../helpers";
import type { DirectoryUser } from "../types";

interface Props {
  users: DirectoryUser[];
  error: string | null;
  onSignIn: (userID: string) => Promise<void>;
}

export function SignIn({ users, error, onSignIn }: Props) {
  return (
    <main className="sign-in">
      <section className="sign-in-card">
        <div className="brand brand--login">
          <span className="brand-mark">FD</span>
          <div>
            <strong>forwarding digital</strong>
            <small>Partner identity provider</small>
          </div>
        </div>
        <p className="eyebrow">Multi-tenant integration</p>
        <h1>Choose a Forwarding Digital user</h1>
        <p>
          These are partner-owned identities. No AutoGPT sign-in is presented.
        </p>
        {error && <div className="error-banner">{error}</div>}
        <div className="user-list">
          {users.map((user) => (
            <button key={user.id} onClick={() => void onSignIn(user.id)}>
              <span>{initials(user.name)}</span>
              <div>
                <strong>{user.name}</strong>
                <small>{user.organizations.join(" · ")}</small>
              </div>
              <em>Continue</em>
            </button>
          ))}
        </div>
      </section>
    </main>
  );
}
