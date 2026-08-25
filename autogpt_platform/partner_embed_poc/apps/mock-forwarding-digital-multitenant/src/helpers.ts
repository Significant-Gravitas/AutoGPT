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
