import { logoID } from "../../helpers";
import {
  actionLabel,
  providerDisplayName,
  serviceFor,
  type ConnectableService,
  type ExpertNeed,
  type ServiceAction,
} from "../../listHelpers";

export type NeedAction =
  | { kind: "allow"; credentialID: string }
  | { kind: "connect"; service: ConnectableService };

/** One "also needs" row, ready to show. */
export interface NeedRow {
  key: string;
  name: string;
  detail: string;
  logoID: string | null;
  /** Null while the list of services is still loading. */
  label: string | null;
  action: NeedAction | null;
}

const WEB: ServiceAction = { kind: "web" };

function canMeet(action: ServiceAction, providers: string[]) {
  return action.kind === "oauth" || action.kind === "sheet"
    ? providers.includes(action.provider)
    : true;
}

function needDetail(need: ExpertNeed) {
  return need.workflowNames.length > 0 ? need.workflowNames.join(", ") : "A scheduled workflow";
}

interface Args {
  needs: ExpertNeed[];
  /** Null while the list of services is still loading. */
  services: ConnectableService[] | null;
  grantedIDs: string[];
}

/**
 * Rows for what the expert's workflows still need. An account allowed since
 * the list was fetched drops out at once; a service the app doesn't know
 * goes to the web.
 */
export function needRows({ needs, services, grantedIDs }: Args): NeedRow[] {
  return needs.flatMap((need): NeedRow[] => {
    if (need.credentialID && grantedIDs.includes(need.credentialID)) return [];
    const service = services ? serviceFor(services, need.providers) : null;
    const name = service?.name ?? providerDisplayName(need.providers[0] ?? "");
    const base = {
      key: need.key,
      name: name || "A service",
      detail: needDetail(need),
      logoID: logoID(service?.key ?? need.providers[0] ?? ""),
    };
    if (need.resolution === "allow" && need.credentialID) {
      return [{ ...base, label: "Allow", action: { kind: "allow", credentialID: need.credentialID } }];
    }
    if (!services) return [{ ...base, label: null, action: null }];
    const known = service ?? { key: need.key, name: base.name, description: null, providers: need.providers, action: WEB };
    // A service row signs in to its lead provider; when that one can't meet
    // this need, the web, which lists every provider, does it instead.
    const action = canMeet(known.action, need.providers) ? known.action : WEB;
    return [
      {
        ...base,
        label: actionLabel(action),
        action: { kind: "connect", service: { ...known, key: need.key, action } },
      },
    ];
  });
}
