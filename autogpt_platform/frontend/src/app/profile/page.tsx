"use client";

import { useSupabase } from "@/components/SupabaseProvider";
import { Button } from "@/components/ui/button";
import useUser from "@/hooks/useUser";
import { useRouter } from "next/navigation";
import { useCallback, useContext, useEffect, useMemo } from "react";
import { FaSpinner } from "react-icons/fa";
import {
  CredentialsProviderData,
  CredentialsProvidersContext,
} from "@/components/integrations/credentials-provider";
import { Separator } from "@/components/ui/separator";
import AutoGPTServerAPI from "@/lib/autogpt-server-api";
import { useToast } from "@/components/ui/use-toast";
import { Alert, AlertDescription } from "@/components/ui/alert";

export default function PrivatePage() {
  const { user, isLoading, error } = useUser();
  const { supabase } = useSupabase();
  const router = useRouter();
  const providers = useContext(CredentialsProvidersContext);
  const api = useMemo(() => new AutoGPTServerAPI(), []);
  const { toast } = useToast();

  const removeCredentials = useCallback(
    async (provider: string, id: string) => {
      try {
        const response = await api.deleteCredentials(provider, id);
        console.log("response", response);
        toast({
          title: "Credentials deleted",
          duration: 2000,
        });
      } catch (error) {
        toast({
          title: "Something went wrong when deleting credentials " + error,
          variant: "destructive",
          duration: 2000,
        });
      }
    },
    [api],
  );

  if (isLoading || !providers || !providers) {
    return (
      <div className="flex h-[80vh] items-center justify-center">
        <FaSpinner className="mr-2 h-16 w-16 animate-spin" />
      </div>
    );
  }

  if (error || !user || !supabase) {
    router.push("/login");
    return null;
  }

  //TODO: Remove this once we have more providers
  delete providers["notion"];
  delete providers["google"];

  return (
    <div>
      <p>Hello {user.email}</p>
      <Button onClick={() => supabase.auth.signOut()}>Log out</Button>
      <div>
        {/* <Alert className="mb-2 mt-2">
          <AlertDescription>Heads up!</AlertDescription>
          <AlertDescription>
            <p>
              You need to manually remove credentials from the Notion after
              deleting them here, see{" "}
            </p>
            <a href="https://www.notion.so/help/add-and-manage-connections-with-the-api#manage-connections-in-your-workspace">
              Notion documentation
            </a>
          </AlertDescription>
        </Alert> */}
        {Object.entries(providers).map(([providerName, provider]) => {
          return (
            <div key={provider.provider} className="mh-2">
              <Separator />
              <div className="text-xl">{provider.providerName}</div>
              {provider.savedApiKeys.length > 0 && (
                <div>
                  <div className="text-md">API Keys</div>
                  {provider.savedApiKeys.map((apiKey) => (
                    <div key={apiKey.id} className="flex flex-row">
                      <p className="p-2">
                        {apiKey.id} - {apiKey.title}
                      </p>
                      <Button
                        variant="destructive"
                        onClick={() =>
                          removeCredentials(providerName, apiKey.id)
                        }
                      >
                        Delete
                      </Button>
                    </div>
                  ))}
                </div>
              )}
              {provider.savedOAuthCredentials.length > 0 && (
                <div>
                  <div className="text-md">OAuth Credentials</div>
                  {provider.savedOAuthCredentials.map((oauth) => (
                    <div key={oauth.id} className="flex flex-row">
                      <p className="p-2">
                        {oauth.id} - {oauth.title}
                      </p>
                      <Button
                        variant="destructive"
                        onClick={() =>
                          removeCredentials(providerName, oauth.id)
                        }
                      >
                        Delete
                      </Button>
                    </div>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
