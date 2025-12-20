import { Metadata } from "next/types";
import { Text } from "@/components/atoms/Text/Text";
import { OAuthAppsSection } from "./components/OAuthAppsSection";

export const metadata: Metadata = { title: "OAuth Apps - AutoGPT Platform" };

const OAuthAppsPage = () => {
  return (
    <div className="container space-y-6 py-10">
      <div className="flex flex-col gap-2">
        <Text variant="h3">OAuth Applications</Text>
        <Text variant="large">
          Manage your OAuth applications that use the AutoGPT Platform API
        </Text>
      </div>
      <OAuthAppsSection />
    </div>
  );
};

export default OAuthAppsPage;
