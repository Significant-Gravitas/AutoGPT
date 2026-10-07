import { AdminImpersonationPanel } from "../components/AdminImpersonationPanel";
import { Text } from "@/components/atoms/Text/Text";

export default function AdminImpersonationPage() {
  return (
    <div className="container mx-auto space-y-6 py-6">
      <div className="space-y-2">
        <Text variant="h3" as="h1">
          User Impersonation
        </Text>
        <Text variant="body" tone="secondary">
          Manage admin user impersonation for debugging and support purposes
        </Text>
      </div>

      <AdminImpersonationPanel />
    </div>
  );
}
