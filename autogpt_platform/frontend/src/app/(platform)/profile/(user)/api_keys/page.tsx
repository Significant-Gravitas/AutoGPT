import { Metadata } from "next/types";
import { APIKeysSection } from "@/components/agptui/composite/APIKeySection";

export const metadata: Metadata = { title: "API Keys - AutoGPT Platform" };

const ApiKeysPage = () => {
  return (
    <div className="w-full pr-4 pt-24 md:pt-0">
      <APIKeysSection />
    </div>
  );
};

export default ApiKeysPage;
