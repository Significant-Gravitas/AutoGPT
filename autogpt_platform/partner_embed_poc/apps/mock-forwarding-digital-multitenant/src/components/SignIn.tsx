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
          <span className="brand-mark">RF</span>
          <div>
            <strong>Relay Freight OS</strong>
            <small>Secure partner workspace</small>
          </div>
        </div>
        <p className="eyebrow">Freight Operations Platform</p>
        <h1>Choose Your Demo Workspace</h1>
        <p>
          Continue with an existing host identity. The embedded assistant never
          asks users to create a second account.
        </p>
        {error && <div className="error-banner">{error}</div>}
        <div className="user-list">
          {users.map((user) => (
            <button
              key={user.id}
              type="button"
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
      </section>
    </main>
  );
}
