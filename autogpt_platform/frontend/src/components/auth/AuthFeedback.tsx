import { AlertCircle, CheckCircle } from "lucide-react";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { HelpItem } from "@/components/auth/help-item";
import { BehaveAs } from "@/lib/utils";

interface Props {
  type: "login" | "signup";
  message?: string | null;
  isError?: boolean;
  behaveAs?: BehaveAs;
}

export default function AuthFeedback({
  type,
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
      {isError &&
        behaveAs === BehaveAs.CLOUD &&
        (type === "signup" ? (
          <Card className="overflow-hidden rounded-lg border border-slate-200 bg-white shadow-sm">
            <CardContent className="p-0">
              <div className="divide-y divide-slate-100">
                <span className="my-3 block text-center text-sm font-medium text-red-500">
                  The provided email may not be allowed to sign up.
                </span>
                <HelpItem
                  title="AutoGPT Platform is currently in closed beta. "
                  description="You can join "
                  linkText="the waitlist here"
                  href="https://agpt.co/waitlist"
                />
                <HelpItem title="Make sure you use the same email address you used to sign up for the waitlist." />
                <HelpItem
                  title="You can self host the platform!"
                  description="Visit our"
                  linkText="GitHub repository"
                  href="https://github.com/Significant-Gravitas/AutoGPT"
                />
              </div>
            </CardContent>
          </Card>
        ) : (
          <Card className="overflow-hidden rounded-lg border border-slate-200 bg-white shadow-sm">
            <CardContent className="p-0">
              <div className="divide-y divide-slate-100">
                <HelpItem
                  title="Having trouble logging in?"
                  description="Make sure you've already "
                  linkText="signed up"
                  href="/signup"
                />
              </div>
            </CardContent>
          </Card>
        ))}

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
