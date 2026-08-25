export interface PartnerEmbedConfig {
  partnerID: string;
  issuer: string;
  jwksURL: string;
  audience: string;
  algorithms: string[];
  allowedCapabilities: string[];
}

export interface VerifiedPartnerIdentity {
  partnerID: string;
  externalSubject: string;
  externalAccountID: string;
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
