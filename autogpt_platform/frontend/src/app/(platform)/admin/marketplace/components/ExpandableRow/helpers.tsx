import { Badge } from "@/components/atoms/Badge/Badge";
import { SubmissionStatus } from "@/lib/autogpt-server-api/types";

export function getStatusBadge(status: SubmissionStatus) {
  switch (status) {
    case SubmissionStatus.PENDING:
      return <Badge variant="warning">Pending</Badge>;
    case SubmissionStatus.APPROVED:
      return <Badge variant="success">Approved</Badge>;
    case SubmissionStatus.REJECTED:
      return <Badge variant="error">Rejected</Badge>;
    default:
      return <Badge variant="info">Draft</Badge>;
  }
}
