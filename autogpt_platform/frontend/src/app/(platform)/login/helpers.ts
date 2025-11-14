export function computeReturnURL(returnUrl: string | null, result: any) {
  return returnUrl
    ? returnUrl
    : (result?.next as string) || (result?.onboarding ? "/onboarding" : "/");
}
