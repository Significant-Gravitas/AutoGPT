import { Badge } from "@/components/atoms/Badge/Badge";

export function SourceBadge({ source }: { source: string }) {
  const variant =
    source === "LOCAL" ? "info" : source === "REMOTE" ? "success" : "error";
  // SEED/REMOTE/LOCAL — LOCAL means admin-owned; the catalog importer
  // never touches it.
  return (
    <Badge variant={source === "SEED" ? "info" : variant} size="small">
      {source}
    </Badge>
  );
}

export function VisibilityBadge({ visibility }: { visibility: string }) {
  return (
    <Badge variant={visibility === "GA" ? "success" : "info"} size="small">
      {visibility}
    </Badge>
  );
}

export function EnabledBadge({ enabled }: { enabled: boolean }) {
  return (
    <Badge variant={enabled ? "success" : "error"} size="small">
      {enabled ? "enabled" : "disabled"}
    </Badge>
  );
}
