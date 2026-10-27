import { useQuery } from "@tanstack/react-query";
import { postV1VerifyMicrosoftConnectionCapabilities } from "@/app/api/__generated__/endpoints/integrations/integrations";

export function getMicrosoftAccessQueryKey(credentialID: string) {
  return ["microsoft-access", credentialID] as const;
}

export function useMicrosoftAccessCheck(credentialID?: string) {
  return useQuery({
    queryKey: getMicrosoftAccessQueryKey(credentialID ?? ""),
    enabled: Boolean(credentialID),
    queryFn: async () => {
      const response = await postV1VerifyMicrosoftConnectionCapabilities({
        credential_id: credentialID ?? "",
      });
      return response.status === 200 ? response.data : null;
    },
  });
}
