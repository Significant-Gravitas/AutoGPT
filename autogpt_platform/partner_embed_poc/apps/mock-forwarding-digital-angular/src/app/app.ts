import {
  Component,
  CUSTOM_ELEMENTS_SCHEMA,
  OnInit,
  signal,
} from "@angular/core";

import type { DirectoryUser, Session, TokenResponse } from "./types";

@Component({
  selector: "fd-root",
  templateUrl: "./app.html",
  styleUrl: "./app.css",
  schemas: [CUSTOM_ELEMENTS_SCHEMA],
})
export class App implements OnInit {
  protected readonly directory = signal<DirectoryUser[]>([]);
  protected readonly session = signal<Session | null>(null);
  protected readonly loading = signal(true);
  protected readonly busy = signal(false);
  protected readonly error = signal<string | null>(null);
  protected readonly accessTokenProvider = () => this.getAccessToken();

  async ngOnInit() {
    await this.initialize();
  }

  protected async signIn(userID: string) {
    this.busy.set(true);
    this.error.set(null);
    try {
      const response = await fetch("/api/session", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ userID }),
      });
      if (!response.ok) throw new Error("Unable to sign in");
      this.session.set((await response.json()) as Session);
    } catch {
      this.error.set("Unable to create the Forwarding Digital session.");
    } finally {
      this.busy.set(false);
    }
  }

  protected async signOut() {
    await fetch("/api/session", { method: "DELETE" });
    this.session.set(null);
  }

  protected async switchOrganization(organizationID: string) {
    this.busy.set(true);
    this.error.set(null);
    try {
      const response = await fetch("/api/session/organization", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ organizationID }),
      });
      if (!response.ok) throw new Error("Organization is not permitted");
      this.session.set((await response.json()) as Session);
    } catch {
      this.error.set("You do not have access to that organization.");
    } finally {
      this.busy.set(false);
    }
  }

  protected async syncTenant() {
    this.busy.set(true);
    this.error.set(null);
    try {
      const response = await fetch("/api/autogpt/sync", { method: "POST" });
      if (!response.ok) throw new Error("Tenant sync failed");
      await this.refreshSession();
    } catch {
      this.error.set("The AutoGPT tenant sync failed.");
    } finally {
      this.busy.set(false);
    }
  }

  protected jobsFor(organizationID: string) {
    if (organizationID === "fd-account-77") {
      return [
        {
          reference: "NSF-1042",
          lane: "Shanghai to Felixstowe",
          status: "Customs hold",
        },
        {
          reference: "NSF-1078",
          lane: "Ningbo to Southampton",
          status: "Docs complete",
        },
      ];
    }
    return [
      {
        reference: "HBR-2208",
        lane: "Rotterdam to Immingham",
        status: "Rail slot pending",
      },
      {
        reference: "HBR-2231",
        lane: "Antwerp to Hull",
        status: "On schedule",
      },
    ];
  }

  private async initialize() {
    try {
      const [directoryResponse, sessionResponse] = await Promise.all([
        fetch("/api/directory"),
        fetch("/api/session"),
      ]);
      if (directoryResponse.ok) {
        const body = (await directoryResponse.json()) as {
          users: DirectoryUser[];
        };
        this.directory.set(body.users);
      }
      if (sessionResponse.ok) {
        this.session.set((await sessionResponse.json()) as Session);
      }
    } catch {
      this.error.set("Unable to load the partner demo.");
    } finally {
      this.loading.set(false);
    }
  }

  private async refreshSession() {
    const response = await fetch("/api/session");
    if (response.ok) this.session.set((await response.json()) as Session);
  }

  private async getAccessToken() {
    const response = await fetch("/api/autogpt/token", { method: "POST" });
    if (!response.ok) throw new Error("Unable to authorize the assistant");
    const body = (await response.json()) as TokenResponse;
    await this.refreshSession();
    return body.access_token;
  }
}
