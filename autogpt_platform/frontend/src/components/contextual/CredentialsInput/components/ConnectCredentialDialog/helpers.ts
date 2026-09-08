export interface ExistingCredential {
  id: string;
  title: string;
  type: string;
}

/** Accounts the user already has that could satisfy the request without a
 *  new sign-in — an expert chat offers these before the connect methods. */
export interface ExistingCredentialsOffer {
  credentials: ExistingCredential[];
  /** Resolves true once the credential is usable; false keeps the dialog
   *  open so the user can retry or add a new one instead. */
  onUse: (credential: ExistingCredential) => Promise<boolean>;
  isPending: boolean;
  error: string | null;
}
