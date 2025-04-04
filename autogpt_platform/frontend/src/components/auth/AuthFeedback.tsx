import { AlertCircle, CheckCircle } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { HelpItem } from "@/components/auth/help-item";
import Link from "next/link";
import { BehaveAs } from "@/lib/utils";

interface Props {
  message?: string | null;
  isError?: boolean;
  behaveAs?: BehaveAs;
}

export default function AuthFeedback({
  message = "",
  isError = false,
  behaveAs = BehaveAs.CLOUD,
}: Props) {
  // If there's no message but isError is true, show a default error message
  const displayMessage =
    message || (isError ? "Something went wrong. Please try again." : "");

  return (
    <div className="mt-4 space-y-4">
      {/* Message feedback */}
      {displayMessage && (
        <div className="text-center text-sm font-medium leading-normal">
          {isError ? (
            <div className="flex items-center justify-center space-x-2 text-red-500">
              <AlertCircle className="h-4 w-4" />
              <span>{displayMessage}</span>
            </div>
          ) : (
            <div className="flex items-center justify-center space-x-2 text-green-600">
              <CheckCircle className="h-4 w-4" />
              <span>{displayMessage}</span>
            </div>
          )}
        </div>
      )}

      {/* Cloud-specific help */}
      {isError && behaveAs === BehaveAs.CLOUD && (
        <div className="mt-2 space-y-2 text-sm">
          <span className="block text-center font-medium text-red-500">
            The provided email may not be allowed to sign up.
          </span>
          <ul className="space-y-2 text-slate-700">
            <li className="flex items-start">
              <span className="mr-2">-</span>
              <span>
                AutoGPT Platform is currently in closed beta. You can join{" "}
                <Link
                  href="https://agpt.co/waitlist"
                  className="font-medium text-slate-950 underline hover:text-slate-700"
                >
                  the waitlist here
                </Link>
                .
              </span>
            </li>
            <li className="flex items-start">
              <span className="mr-2">-</span>
              <span>
                Make sure you use the same email address you used to sign up for
                the waitlist.
              </span>
            </li>
            <li className="flex items-start">
              <span className="mr-2">-</span>
              <span>
                You can self host the platform, visit our{" "}
                <Link
                  href="https://github.com/Significant-Gravitas/AutoGPT"
                  className="font-medium text-slate-950 underline hover:text-slate-700"
                >
                  GitHub repository
                </Link>
                .
              </span>
            </li>
          </ul>
        </div>
      )}

      {/* Local-specific help */}
      {isError && behaveAs === BehaveAs.LOCAL && (
        <Card className="overflow-hidden rounded-lg border border-slate-200 bg-white shadow-sm">
          <CardContent className="p-0">
            <div className="space-y-4 divide-y divide-slate-100">
              <HelpItem
                title="Having trouble getting AutoGPT running locally?"
                description="Ask for help on our"
                linkText="Discord"
                href="https://discord.gg/autogpt"
              />

              <HelpItem
                title="Think you've found a bug?"
                description="Open an issue on our"
                linkText="GitHub"
                href="https://github.com/Significant-Gravitas/AutoGPT"
              />

              <HelpItem
                title="Interested in the cloud-hosted version?"
                description="Join our"
                linkText="waitlist here"
                href="https://agpt.co/waitlist"
              />
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
