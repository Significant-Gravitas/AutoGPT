import {
  Component,
  CUSTOM_ELEMENTS_SCHEMA,
  HostListener,
  OnInit,
  signal,
} from "@angular/core";

import type { DirectoryUser, Session, TokenResponse } from "./types";

type PageID = "overview" | "shipments" | "documents" | "automations";

interface FreightJob {
  reference: string;
  lane: string;
  customer: string;
  eta: string;
  mode: string;
  status: string;
}

interface DocumentSource {
  reference: string;
  status: string;
}

const documentKinds = [
  { label: "Arrival notice", type: "Customer document" },
  { label: "Bill of lading", type: "Carrier document" },
  { label: "Customs pack", type: "Compliance bundle" },
];

export function documentsForJobs(jobs: DocumentSource[]) {
  return jobs.map((job, index) => ({
    name: `${documentKinds[index % documentKinds.length].label} · ${job.reference}`,
    type: documentKinds[index % documentKinds.length].type,
    state: /hold|pending|exception|missing|due/i.test(job.status)
      ? "Needs review"
      : "Verified",
  }));
}

export function assistantNoticeFor(href: string) {
  if (/^\/library\/agents\/[^/]+$/.test(href)) {
    return "Saved agent is ready in the automation library.";
  }
  return "Saved resource is ready in the automation library.";
}

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
  protected readonly accessRequired = signal(false);
  protected readonly accessCode = signal("");
  protected readonly error = signal<string | null>(null);
  protected readonly activePage = signal<PageID>(
    this.pageFromHash(window.location.hash),
  );
  protected readonly assistantOpen = signal(false);
  protected readonly assistantNotice = signal<string | null>(null);
  protected readonly accessTokenProvider = () => this.getAccessToken();
  protected readonly navigation: { id: PageID; label: string }[] = [
    { id: "overview", label: "Operations" },
    { id: "shipments", label: "Shipments" },
    { id: "documents", label: "Documents" },
    { id: "automations", label: "Automations" },
  ];

  async ngOnInit() {
    await this.initialize();
  }

  @HostListener("window:hashchange")
  protected onHashChange() {
    this.activePage.set(this.pageFromHash(window.location.hash));
  }

  protected navigate(page: PageID) {
    window.location.hash = page;
    this.activePage.set(page);
    this.assistantNotice.set(null);
  }

  protected openAssistant() {
    this.assistantOpen.set(true);
  }

  protected closeAssistant() {
    this.assistantOpen.set(false);
  }

  protected handleAssistantNavigation(event: Event) {
    const href = (event as CustomEvent<{ href?: string }>).detail?.href ?? "";
    this.assistantNotice.set(assistantNoticeFor(href));
    this.assistantOpen.set(false);
    window.location.hash = "automations";
    this.activePage.set("automations");
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
      this.error.set("Portside Cloud could not start this workspace session.");
    } finally {
      this.busy.set(false);
    }
  }

  protected async unlockDemo(code: string) {
    this.busy.set(true);
    this.error.set(null);
    try {
      const response = await fetch("/api/demo-access", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ code }),
      });
      if (!response.ok) {
        this.error.set("That team access code was not accepted.");
        return;
      }
      this.accessRequired.set(false);
      await this.loadWorkspace();
      window.requestAnimationFrame(() => window.scrollTo(0, 0));
    } catch {
      this.error.set("Portside Cloud could not verify demo access.");
    } finally {
      this.busy.set(false);
    }
  }

  protected async signOut() {
    await fetch("/api/session", { method: "DELETE" });
    this.session.set(null);
    this.assistantOpen.set(false);
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
      this.assistantNotice.set(null);
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
      this.error.set("The secure AI workspace sync failed. Try again.");
    } finally {
      this.busy.set(false);
    }
  }

  protected jobsFor(organizationID: string): FreightJob[] {
    if (organizationID === "fd-account-77") {
      return [
        {
          reference: "NSF-1042",
          lane: "Shanghai → Felixstowe",
          customer: "Hawthorn Retail",
          eta: "2026-08-26T08:20:00Z",
          mode: "Ocean",
          status: "Customs hold",
        },
        {
          reference: "NSF-1078",
          lane: "Ningbo → Southampton",
          customer: "Calder Foods",
          eta: "2026-08-28T14:10:00Z",
          mode: "Ocean",
          status: "Docs complete",
        },
        {
          reference: "NSF-1096",
          lane: "Chicago → Manchester",
          customer: "Mason Industrial",
          eta: "2026-08-29T07:30:00Z",
          mode: "Air",
          status: "On schedule",
        },
      ];
    }
    return [
      {
        reference: "HBR-2208",
        lane: "Rotterdam → Immingham",
        customer: "Seabrook Components",
        eta: "2026-08-26T11:40:00Z",
        mode: "Rail",
        status: "Rail slot pending",
      },
      {
        reference: "HBR-2231",
        lane: "Antwerp → Hull",
        customer: "Redcliffe Home",
        eta: "2026-08-30T16:00:00Z",
        mode: "Ocean",
        status: "On schedule",
      },
    ];
  }

  protected documentsFor(organizationID: string) {
    return documentsForJobs(this.jobsFor(organizationID));
  }

  protected formatETA(value: string) {
    return new Intl.DateTimeFormat("en-GB", {
      day: "numeric",
      month: "short",
      timeZone: "UTC",
    }).format(new Date(value));
  }

  protected statusKind(status: string) {
    if (/hold|pending|missing/i.test(status)) return "danger";
    if (/due|review/i.test(status)) return "warning";
    return "good";
  }

  protected pageEyebrow(page: PageID) {
    if (page === "shipments") return "Shipment Operations";
    if (page === "documents") return "Document Control";
    if (page === "automations") return "AI & Automation";
    return "Network Operations";
  }

  protected pageTitle(page: PageID) {
    if (page === "shipments") return "Shipments";
    if (page === "documents") return "Documents";
    if (page === "automations") return "Automations";
    return "Operations Desk";
  }

  protected canManageAgents(current: Session) {
    return (
      current.activeOrganization.tools.includes("agents.create") &&
      current.activeOrganization.tools.includes("agents.schedule")
    );
  }

  protected promptsFor(page: PageID) {
    if (page === "shipments") {
      return [
        "List the next 2 arrivals for this tenant with ETA and exception status.",
        "Compare active lanes and flag the highest operational risk.",
      ];
    }
    if (page === "documents") {
      return [
        "Find jobs with missing documents and produce an exception summary.",
        "Create an arrival-notice checklist for the next eligible ocean job.",
      ];
    }
    if (page === "automations") {
      return [
        "Create and save a calculator agent that adds 10 to one numeric input using only enabled blocks.",
        "List my saved agents and schedules, including the next run time.",
        "Schedule the saved calculator agent for every Monday at 09:00 UTC.",
      ];
    }
    return [
      "Summarize today's exceptions and name the jobs that need attention.",
      "List the next 2 arrivals with ETA and current exception status.",
    ];
  }

  private pageFromHash(hash: string): PageID {
    const candidate = hash.replace("#", "") as PageID;
    return ["overview", "shipments", "documents", "automations"].includes(
      candidate,
    )
      ? candidate
      : "overview";
  }

  private async initialize() {
    try {
      const accessResponse = await fetch("/api/demo-access");
      if (!accessResponse.ok) throw new Error("Demo access check failed");
      const access = (await accessResponse.json()) as {
        required: boolean;
        authorized: boolean;
      };
      if (access.required && !access.authorized) {
        this.accessRequired.set(true);
        return;
      }
      await this.loadWorkspace();
    } catch {
      this.error.set("Unable to load the Portside Cloud demo.");
    } finally {
      this.loading.set(false);
    }
  }

  private async loadWorkspace() {
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
