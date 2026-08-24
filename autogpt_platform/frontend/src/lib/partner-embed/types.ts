export interface PartnerEmbedConfig {
  partnerID: string;
  issuer: string;
  jwksURL: string;
  audience: string;
  algorithms: string[];
}

export interface VerifiedPartnerIdentity {
  partnerID: string;
  externalSubject: string;
  externalAccountID: string;
  email: string;
  displayName: string;
  accountName: string;
  isAdmin: boolean;
  capabilities: string[];
  jwtID: string;
  expiresAt: number;
}

export interface ProvisionedPartnerIdentity {
  userID: string;
  organizationID: string;
  teamID: string;
}
