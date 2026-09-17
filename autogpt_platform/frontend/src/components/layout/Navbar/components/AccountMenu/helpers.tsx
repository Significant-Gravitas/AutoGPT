import { IconType } from "@/components/__legacy__/ui/icons";
import {
  ChartIncreaseIcon,
  CreditCardIcon,
  Logout03Icon,
  NewsIcon,
  QuestionIcon,
  Settings01Icon,
  SlidersHorizontalIcon,
  Upload03Icon,
  UserIcon,
} from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

export function getAccountMenuIcon(icon: IconType) {
  const className = "h-[18px] w-[18px] shrink-0";
  switch (icon) {
    case IconType.Edit:
      return <Icon icon={UserIcon} className={className} />;
    case IconType.LayoutDashboard:
      return <Icon icon={ChartIncreaseIcon} className={className} />;
    case IconType.UploadCloud:
      return <Icon icon={Upload03Icon} className={className} />;
    case IconType.Sliders:
      return <Icon icon={SlidersHorizontalIcon} className={className} />;
    case IconType.Settings:
      return <Icon icon={Settings01Icon} className={className} />;
    case IconType.Billing:
      return <Icon icon={CreditCardIcon} className={className} />;
    case IconType.Help:
      return <Icon icon={QuestionIcon} className={className} />;
    case IconType.WhatsNew:
      return <Icon icon={NewsIcon} className={className} />;
    case IconType.LogOut:
      return <Icon icon={Logout03Icon} className={className} />;
    default:
      return null;
  }
}
