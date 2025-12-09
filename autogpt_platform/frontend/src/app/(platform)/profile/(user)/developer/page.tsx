import { Metadata } from "next/types";
import { OAuthClientSection } from "@/app/(platform)/profile/(user)/developer/components/OAuthClientSection/OAuthClientSection";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/__legacy__/ui/card";
import { OAuthClientModals } from "./components/OAuthClientModals/OAuthClientModals";

export const metadata: Metadata = {
  title: "Developer Settings - AutoGPT Platform",
};

function DeveloperPage() {
  return (
    <div className="w-full pr-4 pt-24 md:pt-0">
      <Card>
        <CardHeader>
          <CardTitle>OAuth Applications</CardTitle>
          <CardDescription>
            Register and manage OAuth clients to integrate third-party
            applications with the AutoGPT Platform. OAuth clients allow external
            applications to access AutoGPT APIs on behalf of users.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <OAuthClientModals />
          <OAuthClientSection />
        </CardContent>
      </Card>
    </div>
  );
}

export default DeveloperPage;
