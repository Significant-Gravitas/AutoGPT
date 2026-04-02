import {
  BrainIcon,
  UsersThreeIcon,
  ChatCircleIcon,
  CodeIcon,
  DatabaseIcon,
  TextTIcon,
  MagnifyingGlassIcon,
  GitBranchIcon,
  CubeIcon,
  ArrowSquareInIcon,
  ArrowSquareOutIcon,
  AddressBookIcon,
  FilmStripIcon,
  CheckSquareIcon,
  MegaphoneIcon,
  BugIcon,
  RobotIcon,
  PlugIcon,
} from "@phosphor-icons/react";
import type { Icon } from "@phosphor-icons/react";

const CATEGORY_ICON_MAP: Record<string, Icon> = {
  AI: BrainIcon,
  SOCIAL: UsersThreeIcon,
  COMMUNICATION: ChatCircleIcon,
  DEVELOPER_TOOLS: CodeIcon,
  DATA: DatabaseIcon,
  TEXT: TextTIcon,
  SEARCH: MagnifyingGlassIcon,
  LOGIC: GitBranchIcon,
  BASIC: CubeIcon,
  INPUT: ArrowSquareInIcon,
  OUTPUT: ArrowSquareOutIcon,
  CRM: AddressBookIcon,
  MULTIMEDIA: FilmStripIcon,
  PRODUCTIVITY: CheckSquareIcon,
  MARKETING: MegaphoneIcon,
  ISSUE_TRACKING: BugIcon,
};

export function getCategoryIcon(categoryName: string): Icon {
  return CATEGORY_ICON_MAP[categoryName] ?? CubeIcon;
}

export function getProviderIconPath(providerName: string): string {
  return `/integrations/${providerName}.png`;
}

export { RobotIcon, PlugIcon };
