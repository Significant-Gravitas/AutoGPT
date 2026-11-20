"use client";

import { Link } from "@/components/atoms/Link/Link";
import { Text } from "@/components/atoms/Text/Text";

// The scheme in mobile-app/app.json.
const APP_URL = "autogpt-app://";

export default function AppVerifiedPage() {
  return (
    <main className="flex min-h-screen items-center justify-center px-4 py-8">
      <div className="flex w-full max-w-md flex-col items-center gap-4 rounded-large border border-border-base bg-surface p-6 text-center">
        <Text variant="h3" as="h1">
          Your email is verified
        </Text>
        <Text variant="body">
          Go back to the AutoGPT app on your phone and tap &quot;I&apos;ve
          opened the link&quot;.
        </Text>
        <Link href={APP_URL} isExternal variant="secondary">
          Open the app
        </Link>
      </div>
    </main>
  );
}
