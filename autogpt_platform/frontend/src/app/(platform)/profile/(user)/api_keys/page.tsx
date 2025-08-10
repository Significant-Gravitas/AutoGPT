import { Metadata } from "next/types";
import { APIKeysSection } from "@/app/(platform)/profile/(user)/api_keys/components/APIKeySection/APIKeySection";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { APIKeysModals } from "./components/APIKeysModals/APIKeysModals";

export const metadata: Metadata = { title: "API Keys - AutoGPT Platform" };

const ApiKeysPage = () => {
  return (
    <div className="w-full pr-4 pt-24 md:pt-0">
      <Card>
        <CardHeader>
          <CardTitle>AutoGPT Platform API Keys</CardTitle>
          <CardDescription>
            Manage your AutoGPT Platform API keys for programmatic access
          </CardDescription>
        </CardHeader>
        <CardContent>
          <APIKeysModals />
          <APIKeysSection />
        </CardContent>
      </Card>
    </div>
  );
};

export default ApiKeysPage;
