export function initials(name: string) {
  return name
    .split(" ")
    .map((part) => part[0])
    .join("")
    .slice(0, 2);
}

export function shortID(value: string) {
  return value.slice(0, 8) + "…" + value.slice(-4);
}

export function assistantNoticeFor(href: string) {
  if (/^\/library\/agents\/[^/]+$/.test(href)) {
    return "Saved agent is ready in this tenant's automation library.";
  }
  return "Saved resource is ready in this tenant's automation library.";
}
export type PromptPageID =
  | "overview"
  | "shipments"
  | "documents"
  | "automations";

const operationalPrompts = {
  overview: [
    "Summarize today's exceptions and name the jobs that need attention.",
    "List the next 2 arrivals with ETA and current exception status.",
  ],
  shipments: [
    "Compare the active shipment lanes and flag the highest operational risk.",
    "List the next 2 arrivals for this tenant. Do not invent data.",
  ],
};

const noCapabilityPrompt =
  "Explain what this role can access and which additional capability an administrator would need to grant.";

export function suggestedPromptsFor(
  page: PromptPageID,
  capabilities: string[],
) {
  const enabled = new Set(capabilities);
  if (page === "overview" || page === "shipments") {
    return enabled.has("jobs.read") || enabled.has("reports.read")
      ? operationalPrompts[page]
      : [noCapabilityPrompt];
  }
  if (page === "documents") {
    const prompts: string[] = [];
    if (enabled.has("documents.read")) {
      prompts.push(
        "Find jobs with missing documents and produce an exception summary.",
      );
    }
    if (enabled.has("documents.write")) {
      prompts.push(
        "Draft and save an arrival-notice checklist for the next eligible ocean shipment.",
      );
    }
    return prompts.length > 0 ? prompts : [noCapabilityPrompt];
  }

  const prompts: string[] = [];
  const hasEnabledBlock = capabilities.some((capability) =>
    capability.startsWith("autogpt:block:"),
  );
  if (enabled.has("agents.create") && hasEnabledBlock) {
    prompts.push(
      "Create and save a calculator agent that adds 10 to one numeric input using only enabled blocks.",
    );
  }
  if (enabled.has("agents.run")) {
    prompts.push(
      "Run the saved calculator agent now and report the returned result.",
    );
  }
  if (enabled.has("agents.schedule")) {
    prompts.push(
      "List existing schedules, then schedule the saved calculator agent for every Monday at 09:00 UTC.",
    );
  }
  if (prompts.length > 0) return prompts;

  if (enabled.has("jobs.read")) {
    prompts.push(
      "Turn the current shipment exceptions into a repeatable checklist a manager could automate.",
    );
  }
  if (enabled.has("documents.read")) {
    prompts.push(
      "Review this tenant's document gaps and outline the safest manual follow-up workflow.",
    );
  }
  return prompts.length > 0 ? prompts : [noCapabilityPrompt];
}
export function agentPermissionMessageFor(capabilities: string[]) {
  const enabled = new Set(capabilities);
  const actions: string[] = [];
  const hasEnabledBlock = capabilities.some((capability) =>
    capability.startsWith("autogpt:block:"),
  );
  if (enabled.has("agents.create") && hasEnabledBlock) actions.push("create");
  if (enabled.has("agents.run")) actions.push("run");
  if (enabled.has("agents.schedule")) actions.push("schedule");
  if (actions.length === 0) {
    return "Your current role can analyze operations but cannot create, run, or schedule agents.";
  }
  if (actions.length === 3) {
    return "Manager controls enabled: create, run, and schedule.";
  }
  const enabledActions =
    actions.length === 1
      ? actions[0]
      : `${actions.slice(0, -1).join(", ")} and ${actions.at(-1)}`;
  return `Agent controls enabled for this role: ${enabledActions}. Other actions remain unavailable.`;
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
