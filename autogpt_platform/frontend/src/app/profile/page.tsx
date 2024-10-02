"use client";

import { useSupabase } from "@/components/SupabaseProvider";
import { Button } from "@/components/ui/button";
import useUser from "@/hooks/useUser";
import { useRouter } from "next/navigation";
import { useCallback, useContext, useMemo } from "react";
import { FaSpinner } from "react-icons/fa";
import { CredentialsProvidersContext } from "@/components/integrations/credentials-provider";
import { Separator } from "@/components/ui/separator";
import AutoGPTServerAPI from "@/lib/autogpt-server-api";
import { useToast } from "@/components/ui/use-toast";

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
      } catch (error: any) {
        if (error.response && error.response.status === 501) {
          toast({
            title: "Credentials deleted from AutoGPT",
            description: `You may also manually remove the connection to AutoGPT at ${provider}!`,
            duration: 3000,
          });
          return;
        }

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

  return (
    <div>
      <p>Hello {user.email}</p>
      <Button onClick={() => supabase.auth.signOut()}>Log out</Button>
      <div>
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
