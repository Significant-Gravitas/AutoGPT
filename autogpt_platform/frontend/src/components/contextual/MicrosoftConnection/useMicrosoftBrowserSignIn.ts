import { useGetV1ListProviders } from "@/app/api/__generated__/endpoints/integrations/integrations";
import { MICROSOFT_PROVIDER } from "@/lib/microsoft-setup";

/** Browser sign-in needs the server's Microsoft app secret. Without it, the
 *  same connection comes from a sign-in code instead. */
export function useMicrosoftBrowserSignIn() {
  const providers = useGetV1ListProviders({
    query: {
      select: (response) => (response.status === 200 ? response.data : []),
    },
  });
  const microsoft = providers.data?.find(
    (item) => item.name === MICROSOFT_PROVIDER,
  );
  return {
    browserAvailable: Boolean(
      microsoft?.supported_auth_types?.includes("oauth2"),
    ),
    isLoading: providers.isLoading,
  };
}
