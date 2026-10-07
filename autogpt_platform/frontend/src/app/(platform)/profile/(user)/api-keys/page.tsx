import { Metadata } from "next/types";
import { APIKeysSection } from "@/app/(platform)/profile/(user)/api-keys/components/APIKeySection/APIKeySection";
import { Card } from "@/components/atoms/Card/Card";
import { Text } from "@/components/atoms/Text/Text";
import { APIKeysModals } from "./components/APIKeysModals/APIKeysModals";

export const metadata: Metadata = { title: "API Keys - AutoGPT Platform" };

const ApiKeysPage = () => {
  return (
    <div className="w-full pt-24 pr-4 md:pt-0">
      <Card>
        <div className="mb-6 flex flex-col space-y-1.5">
          <Text variant="h5" as="h3">
            AutoGPT Platform API Keys
          </Text>
          <Text variant="body" tone="muted">
            Manage your AutoGPT Platform API keys for programmatic access
          </Text>
        </div>
        <APIKeysModals />
        <APIKeysSection />
      </Card>
    </div>
  );
};

export default ApiKeysPage;
