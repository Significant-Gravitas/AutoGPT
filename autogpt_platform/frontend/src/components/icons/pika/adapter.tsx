/**
 * Server-safe icon adapter. Each export renders the mapped Pika (stroke)
 * icon directly — no feature flag, no client hooks — so it works in both
 * server and client components. Call sites keep importing the same Phosphor
 * names; only the import source is swapped to this module. Generated from
 * .context/pika/mapping.json.
 */
import React from "react";
import type { Icon, IconProps } from "@phosphor-icons/react";

import PiActivityStroke from "./vendor/PiActivityStroke";
import PiAlertCircleStroke from "./vendor/PiAlertCircleStroke";
import PiAlertTriangleStroke from "./vendor/PiAlertTriangleStroke";
import PiArrowDownStroke from "./vendor/PiArrowDownStroke";
import PiArrowLeftStroke from "./vendor/PiArrowLeftStroke";
import PiArrowRightStroke from "./vendor/PiArrowRightStroke";
import PiArrowTurnLeftUpStroke from "./vendor/PiArrowTurnLeftUpStroke";
import PiArrowTurnRightDownStroke from "./vendor/PiArrowTurnRightDownStroke";
import PiArrowTurnUpRightStroke from "./vendor/PiArrowTurnUpRightStroke";
import PiArrowUpStroke from "./vendor/PiArrowUpStroke";
import PiAtomStroke from "./vendor/PiAtomStroke";
import PiAutomationStroke from "./vendor/PiAutomationStroke";
import PiBanRightStroke from "./vendor/PiBanRightStroke";
import PiBarchartDefaultStroke from "./vendor/PiBarchartDefaultStroke";
import PiBoldStroke from "./vendor/PiBoldStroke";
import PiBotStroke from "./vendor/PiBotStroke";
import PiBugStroke from "./vendor/PiBugStroke";
import PiBulbDefaultStroke from "./vendor/PiBulbDefaultStroke";
import PiCalendarDefaultStroke from "./vendor/PiCalendarDefaultStroke";
import PiCameraDefaultStroke from "./vendor/PiCameraDefaultStroke";
import PiChatChattingStroke from "./vendor/PiChatChattingStroke";
import PiChatPlusStroke from "./vendor/PiChatPlusStroke";
import PiChatDefaultStroke from "./vendor/PiChatDefaultStroke";
import PiChatTypingStroke from "./vendor/PiChatTypingStroke";
import PiCheckTickCircleStroke from "./vendor/PiCheckTickCircleStroke";
import PiCheckTickSingleStroke from "./vendor/PiCheckTickSingleStroke";
import PiCheckTickSquareStroke from "./vendor/PiCheckTickSquareStroke";
import PiChevronDownStroke from "./vendor/PiChevronDownStroke";
import PiChevronLeftStroke from "./vendor/PiChevronLeftStroke";
import PiChevronRightStroke from "./vendor/PiChevronRightStroke";
import PiChevronSortVerticalStroke from "./vendor/PiChevronSortVerticalStroke";
import PiChevronUpStroke from "./vendor/PiChevronUpStroke";
import PiCircleDashedStroke from "./vendor/PiCircleDashedStroke";
import PiCircleStroke from "./vendor/PiCircleStroke";
import PiClockDefaultStroke from "./vendor/PiClockDefaultStroke";
import PiCloudArrowUploadStroke from "./vendor/PiCloudArrowUploadStroke";
import PiCodeStroke from "./vendor/PiCodeStroke";
import PiCookieStroke from "./vendor/PiCookieStroke";
import PiCopyDefaultStroke from "./vendor/PiCopyDefaultStroke";
import PiCreditCardStroke from "./vendor/PiCreditCardStroke";
import PiCubeStroke from "./vendor/PiCubeStroke";
import PiCurlyBracesCodeDefaultStroke from "./vendor/PiCurlyBracesCodeDefaultStroke";
import PiCurrencySignDollarStroke from "./vendor/PiCurrencySignDollarStroke";
import PiDeleteDustbin01Stroke from "./vendor/PiDeleteDustbin01Stroke";
import PiDiscordStroke from "./vendor/PiDiscordStroke";
import PiDownloadDownStroke from "./vendor/PiDownloadDownStroke";
import PiEarStroke from "./vendor/PiEarStroke";
import PiEnvelopeDefaultStroke from "./vendor/PiEnvelopeDefaultStroke";
import PiExternalLinkSquareStroke from "./vendor/PiExternalLinkSquareStroke";
import PiEyeOffStroke from "./vendor/PiEyeOffStroke";
import PiEyeOnStroke from "./vendor/PiEyeOnStroke";
import PiFaceSadStroke from "./vendor/PiFaceSadStroke";
import PiFacebookStroke from "./vendor/PiFacebookStroke";
import PiFeatherPenStroke from "./vendor/PiFeatherPenStroke";
import PiFileCodeStroke from "./vendor/PiFileCodeStroke";
import PiFileDefaultStroke from "./vendor/PiFileDefaultStroke";
import PiFilePdfFormatStroke from "./vendor/PiFilePdfFormatStroke";
import PiFilePlusStroke from "./vendor/PiFilePlusStroke";
import PiFileSearchStroke from "./vendor/PiFileSearchStroke";
import PiFileTextStroke from "./vendor/PiFileTextStroke";
import PiFilterFunnelStroke from "./vendor/PiFilterFunnelStroke";
import PiFilterHorizontalStroke from "./vendor/PiFilterHorizontalStroke";
import PiFloppyDefaultStroke from "./vendor/PiFloppyDefaultStroke";
import PiFolderDefaultStroke from "./vendor/PiFolderDefaultStroke";
import PiFolderOpenStroke from "./vendor/PiFolderOpenStroke";
import PiFolderPlusStroke from "./vendor/PiFolderPlusStroke";
import PiFrameStroke from "./vendor/PiFrameStroke";
import PiGaugeSpeedometerStroke from "./vendor/PiGaugeSpeedometerStroke";
import PiGitBranchStroke from "./vendor/PiGitBranchStroke";
import PiGithubStroke from "./vendor/PiGithubStroke";
import PiGlobeStroke from "./vendor/PiGlobeStroke";
import PiGoogleStroke from "./vendor/PiGoogleStroke";
import PiGraphChartLineStroke from "./vendor/PiGraphChartLineStroke";
import PiGraphTrendLineUpwardStroke from "./vendor/PiGraphTrendLineUpwardStroke";
import PiGrid01Stroke from "./vendor/PiGrid01Stroke";
import PiGridTableStroke from "./vendor/PiGridTableStroke";
import PiHeadphonesStroke from "./vendor/PiHeadphonesStroke";
import PiHeartStroke from "./vendor/PiHeartStroke";
import PiHomeDefaultStroke from "./vendor/PiHomeDefaultStroke";
import PiHourglassStroke from "./vendor/PiHourglassStroke";
import PiInboxDefaultStroke from "./vendor/PiInboxDefaultStroke";
import PiInformationCircleStroke from "./vendor/PiInformationCircleStroke";
import PiInstagramStroke from "./vendor/PiInstagramStroke";
import PiIphoneStroke from "./vendor/PiIphoneStroke";
import PiItalicStroke from "./vendor/PiItalicStroke";
import PiKeyLeftStroke from "./vendor/PiKeyLeftStroke";
import PiLabFlaskConicalStroke from "./vendor/PiLabFlaskConicalStroke";
import PiLaptopStroke from "./vendor/PiLaptopStroke";
import PiLayersToStroke from "./vendor/PiLayersToStroke";
import PiLayoutGridTwoVerticalStroke from "./vendor/PiLayoutGridTwoVerticalStroke";
import PiLibraryStroke from "./vendor/PiLibraryStroke";
import PiLightningThunderElectricOnStroke from "./vendor/PiLightningThunderElectricOnStroke";
import PiLinkHorizontalBrokenStroke from "./vendor/PiLinkHorizontalBrokenStroke";
import PiLinkHorizontalStroke from "./vendor/PiLinkHorizontalStroke";
import PiLinkedinStroke from "./vendor/PiLinkedinStroke";
import PiListCheckStroke from "./vendor/PiListCheckStroke";
import PiListDefaultStroke from "./vendor/PiListDefaultStroke";
import PiLockCloseStroke from "./vendor/PiLockCloseStroke";
import PiLogInRightStroke from "./vendor/PiLogInRightStroke";
import PiLogOutRightStroke from "./vendor/PiLogOutRightStroke";
import PiMagicWandStroke from "./vendor/PiMagicWandStroke";
import PiMathStroke from "./vendor/PiMathStroke";
import PiMaximizeFourArrowStroke from "./vendor/PiMaximizeFourArrowStroke";
import PiMaximizeLineArrowStroke from "./vendor/PiMaximizeLineArrowStroke";
import PiMediaPlaySquareStroke from "./vendor/PiMediaPlaySquareStroke";
import PiMicMicrophoneStroke from "./vendor/PiMicMicrophoneStroke";
import PiMinusCircleStroke from "./vendor/PiMinusCircleStroke";
import PiMinusDefaultStroke from "./vendor/PiMinusDefaultStroke";
import PiMoneyDollarBagStroke from "./vendor/PiMoneyDollarBagStroke";
import PiMonitor01Stroke from "./vendor/PiMonitor01Stroke";
import PiMoonStroke from "./vendor/PiMoonStroke";
import PiMultipleCrossCancelCircleStroke from "./vendor/PiMultipleCrossCancelCircleStroke";
import PiMultipleCrossCancelDefaultStroke from "./vendor/PiMultipleCrossCancelDefaultStroke";
import PiNotebookStroke from "./vendor/PiNotebookStroke";
import PiNotificationBellOffStroke from "./vendor/PiNotificationBellOffStroke";
import PiNotificationBellOnStroke from "./vendor/PiNotificationBellOnStroke";
import PiNotionStroke from "./vendor/PiNotionStroke";
import PiPackage01Stroke from "./vendor/PiPackage01Stroke";
import PiPaintBrushStroke from "./vendor/PiPaintBrushStroke";
import PiPauseCircleStroke from "./vendor/PiPauseCircleStroke";
import PiPencilEditBoxSolidStroke from "./vendor/PiPencilEditBoxSolidStroke";
import PiPencilEditBoxStroke from "./vendor/PiPencilEditBoxStroke";
import PiPencilEditLineStroke from "./vendor/PiPencilEditLineStroke";
import PiPencilEditStroke from "./vendor/PiPencilEditStroke";
import PiPhoneDefaultStroke from "./vendor/PiPhoneDefaultStroke";
import PiPhoneOutgoingStroke from "./vendor/PiPhoneOutgoingStroke";
import PiPhotoImageDefaultStroke from "./vendor/PiPhotoImageDefaultStroke";
import PiPhotoImageRemoveStroke from "./vendor/PiPhotoImageRemoveStroke";
import PiPinDefaultStroke from "./vendor/PiPinDefaultStroke";
import PiPinSlantStroke from "./vendor/PiPinSlantStroke";
import PiPlayBigStroke from "./vendor/PiPlayBigStroke";
import PiPlayCircleStroke from "./vendor/PiPlayCircleStroke";
import PiPluginAddonDefaultStroke from "./vendor/PiPluginAddonDefaultStroke";
import PiPluginAddonPuzzleStroke from "./vendor/PiPluginAddonPuzzleStroke";
import PiPlusCircleStroke from "./vendor/PiPlusCircleStroke";
import PiPlusDefaultStroke from "./vendor/PiPlusDefaultStroke";
import PiPowerDefaultStroke from "./vendor/PiPowerDefaultStroke";
import PiPresentationBargraphStroke from "./vendor/PiPresentationBargraphStroke";
import PiQuestionMarkCircleStroke from "./vendor/PiQuestionMarkCircleStroke";
import PiReceipt01Stroke from "./vendor/PiReceipt01Stroke";
import PiRefreshStroke from "./vendor/PiRefreshStroke";
import PiReminderAnticlockwiseStroke from "./vendor/PiReminderAnticlockwiseStroke";
import PiReminderClockwiseStroke from "./vendor/PiReminderClockwiseStroke";
import PiRocketShipStroke from "./vendor/PiRocketShipStroke";
import PiRotateLeftStroke from "./vendor/PiRotateLeftStroke";
import PiRotateRightStroke from "./vendor/PiRotateRightStroke";
import PiSearchDefaultStroke from "./vendor/PiSearchDefaultStroke";
import PiSettings01Stroke from "./vendor/PiSettings01Stroke";
import PiShare01Stroke from "./vendor/PiShare01Stroke";
import PiShare02Stroke from "./vendor/PiShare02Stroke";
import PiShieldCheckStroke from "./vendor/PiShieldCheckStroke";
import PiShieldStroke from "./vendor/PiShieldStroke";
import PiSidebarDefaultStroke from "./vendor/PiSidebarDefaultStroke";
import PiSparkleAI01Stroke from "./vendor/PiSparkleAI01Stroke";
import PiSpeakerOnStroke from "./vendor/PiSpeakerOnStroke";
import PiSpinnerStroke from "./vendor/PiSpinnerStroke";
import PiSquareStroke from "./vendor/PiSquareStroke";
import PiStarStroke from "./vendor/PiStarStroke";
import PiStopBigStroke from "./vendor/PiStopBigStroke";
import PiStopCircleStroke from "./vendor/PiStopCircleStroke";
import PiStopSquareStroke from "./vendor/PiStopSquareStroke";
import PiStoreDefaultStroke from "./vendor/PiStoreDefaultStroke";
import PiStrikeThroughStroke from "./vendor/PiStrikeThroughStroke";
import PiSunStroke from "./vendor/PiSunStroke";
import PiTerminalConsoleSquareStroke from "./vendor/PiTerminalConsoleSquareStroke";
import PiThreeDotsMenuHorizontalStroke from "./vendor/PiThreeDotsMenuHorizontalStroke";
import PiThreeDotsMenuVerticalStroke from "./vendor/PiThreeDotsMenuVerticalStroke";
import PiThumbReactionDislikeStroke from "./vendor/PiThumbReactionDislikeStroke";
import PiThumbReactionLikeStroke from "./vendor/PiThumbReactionLikeStroke";
import PiTiktokStroke from "./vendor/PiTiktokStroke";
import PiTimerDefaultStroke from "./vendor/PiTimerDefaultStroke";
import PiTokenStroke from "./vendor/PiTokenStroke";
import PiToolsStroke from "./vendor/PiToolsStroke";
import PiTwitterStroke from "./vendor/PiTwitterStroke";
import PiUfoStroke from "./vendor/PiUfoStroke";
import PiUploadUpStroke from "./vendor/PiUploadUpStroke";
import PiUserCheckStroke from "./vendor/PiUserCheckStroke";
import PiUserCircleDottedStroke from "./vendor/PiUserCircleDottedStroke";
import PiUserCircleStroke from "./vendor/PiUserCircleStroke";
import PiUserDefaultStroke from "./vendor/PiUserDefaultStroke";
import PiUserPlusStroke from "./vendor/PiUserPlusStroke";
import PiUserRemoveStroke from "./vendor/PiUserRemoveStroke";
import PiUserTwoStroke from "./vendor/PiUserTwoStroke";
import PiUturnLeftStroke from "./vendor/PiUturnLeftStroke";
import PiUturnRightStroke from "./vendor/PiUturnRightStroke";
import PiVerificationCheckStroke from "./vendor/PiVerificationCheckStroke";
import PiVideoRecordingStroke from "./vendor/PiVideoRecordingStroke";
import PiWalletDefaultStroke from "./vendor/PiWalletDefaultStroke";
import PiWebhookStroke from "./vendor/PiWebhookStroke";
import PiWindowBrowserStroke from "./vendor/PiWindowBrowserStroke";
import PiXComStroke from "./vendor/PiXComStroke";
import PiYoutubeStroke from "./vendor/PiYoutubeStroke";

interface PikaIconProps extends React.SVGProps<SVGSVGElement> {
  size?: number;
  color?: string;
  className?: string;
  ariaLabel?: string;
}

function makeIcon(PikaIcon: React.ComponentType<PikaIconProps>): Icon {
  // Memoized so a parent re-render with unchanged props skips re-rendering the
  // SVG. `_ref` is accepted (via forwardRef) so callers can pass a ref without
  // a warning, but the Pika vendor icons are plain function components, so it
  // is intentionally not forwarded.
  const AdaptiveIcon = React.forwardRef<SVGSVGElement, IconProps>(
    function AdaptiveIcon(props, _ref) {
      const {
        weight: _weight,
        mirrored: _mirrored,
        alt,
        size,
        color,
        className,
        style,
        ...rest
      } = props;
      const numericSize =
        typeof size === "number"
          ? size
          : size != null
            ? Number.parseFloat(String(size))
            : undefined;
      // Pika vendor icons hardcode role="img" + a default aria-label, which
      // makes otherwise-decorative icons announced content and pollutes the
      // accessible name of any labeled control they sit inside. Phosphor icons
      // are unlabeled by default, so mirror that: strip role/aria-label unless
      // the caller passes `alt` (Phosphor's opt-in labeling prop).
      const a11yProps = alt
        ? { role: "img", "aria-label": alt }
        : { role: undefined, "aria-label": undefined };
      return (
        <PikaIcon
          size={
            numericSize != null && !Number.isNaN(numericSize)
              ? numericSize
              : undefined
          }
          color={color}
          className={className}
          style={style}
          {...a11yProps}
          {...(rest as PikaIconProps)}
        />
      );
    },
  );
  return React.memo(AdaptiveIcon) as unknown as Icon;
}

export type { Icon, IconProps };

export const Alien = makeIcon(PiUfoStroke);
export const AppWindowIcon = makeIcon(PiWindowBrowserStroke);
export const ArrowBendLeftUpIcon = makeIcon(PiArrowTurnLeftUpStroke);
export const ArrowBendRightDownIcon = makeIcon(PiArrowTurnRightDownStroke);
export const ArrowBendUpRight = makeIcon(PiArrowTurnUpRightStroke);
export const ArrowClockwise = makeIcon(PiRotateRightStroke);
export const ArrowCounterClockwise = makeIcon(PiRotateLeftStroke);
export const ArrowDownIcon = makeIcon(PiArrowDownStroke);
export const ArrowLeft = makeIcon(PiArrowLeftStroke);
export const ArrowLeftIcon = makeIcon(PiArrowLeftStroke);
export const ArrowRight = makeIcon(PiArrowRightStroke);
export const ArrowRightIcon = makeIcon(PiArrowRightStroke);
export const ArrowSquareOut = makeIcon(PiExternalLinkSquareStroke);
export const ArrowSquareOutIcon = makeIcon(PiExternalLinkSquareStroke);
export const ArrowUUpLeftIcon = makeIcon(PiUturnLeftStroke);
export const ArrowUUpRightIcon = makeIcon(PiUturnRightStroke);
export const ArrowUp = makeIcon(PiArrowUpStroke);
export const ArrowUpIcon = makeIcon(PiArrowUpStroke);
export const ArrowsClockwiseIcon = makeIcon(PiRefreshStroke);
export const ArrowsOutIcon = makeIcon(PiMaximizeFourArrowStroke);
export const ArrowsOutSimpleIcon = makeIcon(PiMaximizeLineArrowStroke);
export const ArticleIcon = makeIcon(PiFileTextStroke);
export const Bell = makeIcon(PiNotificationBellOnStroke);
export const BellRinging = makeIcon(PiNotificationBellOnStroke);
export const BellRingingIcon = makeIcon(PiNotificationBellOnStroke);
export const BellSimpleRingingIcon = makeIcon(PiNotificationBellOnStroke);
export const BellSlash = makeIcon(PiNotificationBellOffStroke);
export const BookOpenIcon = makeIcon(PiNotebookStroke);
export const Books = makeIcon(PiLibraryStroke);
export const BracketsCurlyIcon = makeIcon(PiCurlyBracesCodeDefaultStroke);
export const Brain = makeIcon(PiAtomStroke);
export const BrainIcon = makeIcon(PiAtomStroke);
export const Bug = makeIcon(PiBugStroke);
export const CalculatorIcon = makeIcon(PiMathStroke);
export const CalendarDotsIcon = makeIcon(PiCalendarDefaultStroke);
export const CameraIcon = makeIcon(PiCameraDefaultStroke);
export const CardsThreeIcon = makeIcon(PiLayoutGridTwoVerticalStroke);
export const CaretDown = makeIcon(PiChevronDownStroke);
export const CaretDownIcon = makeIcon(PiChevronDownStroke);
export const CaretLeft = makeIcon(PiChevronLeftStroke);
export const CaretLeftIcon = makeIcon(PiChevronLeftStroke);
export const CaretRight = makeIcon(PiChevronRightStroke);
export const CaretRightIcon = makeIcon(PiChevronRightStroke);
export const CaretUpDownIcon = makeIcon(PiChevronSortVerticalStroke);
export const CaretUpIcon = makeIcon(PiChevronUpStroke);
export const ChalkboardIcon = makeIcon(PiPresentationBargraphStroke);
export const ChartBarIcon = makeIcon(PiBarchartDefaultStroke);
export const ChartLineIcon = makeIcon(PiGraphChartLineStroke);
export const ChartLineUpIcon = makeIcon(PiGraphTrendLineUpwardStroke);
export const ChatCircle = makeIcon(PiChatDefaultStroke);
export const ChatCircleDotsIcon = makeIcon(PiChatTypingStroke);
export const ChatCircleIcon = makeIcon(PiChatDefaultStroke);
export const ChatCircleTextIcon = makeIcon(PiChatDefaultStroke);
export const ChatTeardropDotsIcon = makeIcon(PiChatTypingStroke);
export const ChatsCircleIcon = makeIcon(PiChatChattingStroke);
export const ChatsIcon = makeIcon(PiChatChattingStroke);
export const Check = makeIcon(PiCheckTickSingleStroke);
export const CheckCircle = makeIcon(PiCheckTickCircleStroke);
export const CheckCircleIcon = makeIcon(PiCheckTickCircleStroke);
export const CheckIcon = makeIcon(PiCheckTickSingleStroke);
export const CheckSquareIcon = makeIcon(PiCheckTickSquareStroke);
export const CircleDashedIcon = makeIcon(PiCircleDashedStroke);
export const CircleIcon = makeIcon(PiCircleStroke);
export const CircleNotch = makeIcon(PiSpinnerStroke);
export const CircleNotchIcon = makeIcon(PiSpinnerStroke);
export const Clock = makeIcon(PiClockDefaultStroke);
export const ClockClockwiseIcon = makeIcon(PiReminderClockwiseStroke);
export const ClockCountdownIcon = makeIcon(PiTimerDefaultStroke);
export const ClockCounterClockwiseIcon = makeIcon(
  PiReminderAnticlockwiseStroke,
);
export const ClockIcon = makeIcon(PiClockDefaultStroke);
export const CloudArrowUp = makeIcon(PiCloudArrowUploadStroke);
export const Code = makeIcon(PiCodeStroke);
export const CodeIcon = makeIcon(PiCodeStroke);
export const CoinIcon = makeIcon(PiTokenStroke);
export const CoinsIcon = makeIcon(PiMoneyDollarBagStroke);
export const CookieIcon = makeIcon(PiCookieStroke);
export const Copy = makeIcon(PiCopyDefaultStroke);
export const CopyIcon = makeIcon(PiCopyDefaultStroke);
export const CopySimple = makeIcon(PiCopyDefaultStroke);
export const Cpu = makeIcon(PiTerminalConsoleSquareStroke);
export const CreditCard = makeIcon(PiCreditCardStroke);
export const CreditCardIcon = makeIcon(PiCreditCardStroke);
export const Cube = makeIcon(PiCubeStroke);
export const CubeIcon = makeIcon(PiCubeStroke);
export const CurrencyDollar = makeIcon(PiCurrencySignDollarStroke);
export const CurrencyDollarSimpleIcon = makeIcon(PiCurrencySignDollarStroke);
export const DeviceMobile = makeIcon(PiIphoneStroke);
export const DeviceMobileIcon = makeIcon(PiIphoneStroke);
export const DiscordLogo = makeIcon(PiDiscordStroke);
export const DiscordLogoIcon = makeIcon(PiDiscordStroke);
export const DotsNineIcon = makeIcon(PiGrid01Stroke);
export const DotsThree = makeIcon(PiThreeDotsMenuHorizontalStroke);
export const DotsThreeIcon = makeIcon(PiThreeDotsMenuHorizontalStroke);
export const DotsThreeOutlineVerticalIcon = makeIcon(
  PiThreeDotsMenuVerticalStroke,
);
export const DotsThreeVertical = makeIcon(PiThreeDotsMenuVerticalStroke);
export const DotsThreeVerticalIcon = makeIcon(PiThreeDotsMenuVerticalStroke);
export const Download = makeIcon(PiDownloadDownStroke);
export const DownloadIcon = makeIcon(PiDownloadDownStroke);
export const DownloadSimple = makeIcon(PiDownloadDownStroke);
export const DownloadSimpleIcon = makeIcon(PiDownloadDownStroke);
export const EarIcon = makeIcon(PiEarStroke);
export const EnvelopeSimpleIcon = makeIcon(PiEnvelopeDefaultStroke);
export const Eye = makeIcon(PiEyeOnStroke);
export const EyeClosedIcon = makeIcon(PiEyeOffStroke);
export const EyeIcon = makeIcon(PiEyeOnStroke);
export const EyeSlash = makeIcon(PiEyeOffStroke);
export const FacebookLogo = makeIcon(PiFacebookStroke);
export const File = makeIcon(PiFileDefaultStroke);
export const FileHtml = makeIcon(PiFileCodeStroke);
export const FileIcon = makeIcon(PiFileDefaultStroke);
export const FileMagnifyingGlassIcon = makeIcon(PiFileSearchStroke);
export const FilePdfIcon = makeIcon(PiFilePdfFormatStroke);
export const FilePlusIcon = makeIcon(PiFilePlusStroke);
export const FileText = makeIcon(PiFileTextStroke);
export const FileTextIcon = makeIcon(PiFileTextStroke);
export const FilesIcon = makeIcon(PiFileDefaultStroke);
export const Flask = makeIcon(PiLabFlaskConicalStroke);
export const FlaskIcon = makeIcon(PiLabFlaskConicalStroke);
export const FloppyDisk = makeIcon(PiFloppyDefaultStroke);
export const FloppyDiskIcon = makeIcon(PiFloppyDefaultStroke);
export const FlowArrow = makeIcon(PiAutomationStroke);
export const FlowArrowIcon = makeIcon(PiAutomationStroke);
export const Folder = makeIcon(PiFolderDefaultStroke);
export const FolderIcon = makeIcon(PiFolderDefaultStroke);
export const FolderOpen = makeIcon(PiFolderOpenStroke);
export const FolderOpenIcon = makeIcon(PiFolderOpenStroke);
export const FolderPlusIcon = makeIcon(PiFolderPlusStroke);
export const FolderSimpleIcon = makeIcon(PiFolderDefaultStroke);
export const FolderSimplePlusIcon = makeIcon(PiFolderPlusStroke);
export const FoldersIcon = makeIcon(PiFolderDefaultStroke);
export const FrameCornersIcon = makeIcon(PiFrameStroke);
export const FunnelIcon = makeIcon(PiFilterFunnelStroke);
export const Gauge = makeIcon(PiGaugeSpeedometerStroke);
export const GaugeIcon = makeIcon(PiGaugeSpeedometerStroke);
export const Gear = makeIcon(PiSettings01Stroke);
export const GearIcon = makeIcon(PiSettings01Stroke);
export const GearSix = makeIcon(PiSettings01Stroke);
export const GithubLogo = makeIcon(PiGithubStroke);
export const GithubLogoIcon = makeIcon(PiGithubStroke);
export const Globe = makeIcon(PiGlobeStroke);
export const GlobeIcon = makeIcon(PiGlobeStroke);
export const GlobeSimple = makeIcon(PiGlobeStroke);
export const GoogleLogoIcon = makeIcon(PiGoogleStroke);
export const HammerIcon = makeIcon(PiToolsStroke);
export const HeadsetIcon = makeIcon(PiHeadphonesStroke);
export const Heart = makeIcon(PiHeartStroke);
export const HeartIcon = makeIcon(PiHeartStroke);
export const Heartbeat = makeIcon(PiActivityStroke);
export const HourglassIcon = makeIcon(PiHourglassStroke);
export const HouseIcon = makeIcon(PiHomeDefaultStroke);
export const Image = makeIcon(PiPhotoImageDefaultStroke);
export const ImageBroken = makeIcon(PiPhotoImageRemoveStroke);
export const ImageBrokenIcon = makeIcon(PiPhotoImageRemoveStroke);
export const ImageIcon = makeIcon(PiPhotoImageDefaultStroke);
export const ImagesIcon = makeIcon(PiPhotoImageDefaultStroke);
export const Info = makeIcon(PiInformationCircleStroke);
export const InfoIcon = makeIcon(PiInformationCircleStroke);
export const InstagramLogo = makeIcon(PiInstagramStroke);
export const Key = makeIcon(PiKeyLeftStroke);
export const KeyIcon = makeIcon(PiKeyLeftStroke);
export const KeyholeIcon = makeIcon(PiLockCloseStroke);
export const Laptop = makeIcon(PiLaptopStroke);
export const LegoIcon = makeIcon(PiPluginAddonPuzzleStroke);
export const LightbulbIcon = makeIcon(PiBulbDefaultStroke);
export const Lightning = makeIcon(PiLightningThunderElectricOnStroke);
export const LightningIcon = makeIcon(PiLightningThunderElectricOnStroke);
export const LinkBreak = makeIcon(PiLinkHorizontalBrokenStroke);
export const LinkIcon = makeIcon(PiLinkHorizontalStroke);
export const LinkSimpleIcon = makeIcon(PiLinkHorizontalStroke);
export const LinkedinLogo = makeIcon(PiLinkedinStroke);
export const List = makeIcon(PiListDefaultStroke);
export const ListBulletsIcon = makeIcon(PiListDefaultStroke);
export const ListChecksIcon = makeIcon(PiListCheckStroke);
export const ListIcon = makeIcon(PiListDefaultStroke);
export const Lock = makeIcon(PiLockCloseStroke);
export const MagicWandIcon = makeIcon(PiMagicWandStroke);
export const MagnifyingGlass = makeIcon(PiSearchDefaultStroke);
export const MagnifyingGlassIcon = makeIcon(PiSearchDefaultStroke);
export const MediumLogoIcon = makeIcon(PiFeatherPenStroke);
export const MicrophoneIcon = makeIcon(PiMicMicrophoneStroke);
export const MinusCircleIcon = makeIcon(PiMinusCircleStroke);
export const MinusIcon = makeIcon(PiMinusDefaultStroke);
export const MonitorIcon = makeIcon(PiMonitor01Stroke);
export const MonitorPlayIcon = makeIcon(PiMediaPlaySquareStroke);
export const MoonIcon = makeIcon(PiMoonStroke);
export const NotePencilIcon = makeIcon(PiPencilEditBoxStroke);
// "New Task" sidebar action uses the filled pencil variant; Phosphor has no
// solid-stroke equivalent, so this name exists only in the adapter.
export const NotePencilSolidIcon = makeIcon(PiPencilEditBoxSolidStroke);
// "New chat" compose icon: Phosphor has no chat-plus equivalent, so this name
// exists only in the adapter.
export const ChatPlusIcon = makeIcon(PiChatPlusStroke);
export const NotionLogoIcon = makeIcon(PiNotionStroke);
export const Package = makeIcon(PiPackage01Stroke);
export const PackageIcon = makeIcon(PiPackage01Stroke);
export const PaintBrushIcon = makeIcon(PiPaintBrushStroke);
export const Password = makeIcon(PiLockCloseStroke);
export const PauseCircleIcon = makeIcon(PiPauseCircleStroke);
export const PencilIcon = makeIcon(PiPencilEditStroke);
export const PencilLineIcon = makeIcon(PiPencilEditLineStroke);
export const PencilSimple = makeIcon(PiPencilEditStroke);
export const PencilSimpleIcon = makeIcon(PiPencilEditStroke);
export const PhoneCallIcon = makeIcon(PiPhoneOutgoingStroke);
export const PhoneIcon = makeIcon(PiPhoneDefaultStroke);
export const Play = makeIcon(PiPlayBigStroke);
export const PlayCircleIcon = makeIcon(PiPlayCircleStroke);
export const PlayIcon = makeIcon(PiPlayBigStroke);
export const PlugIcon = makeIcon(PiPluginAddonDefaultStroke);
export const PlugsConnectedIcon = makeIcon(PiPluginAddonDefaultStroke);
export const PlugsIcon = makeIcon(PiPluginAddonDefaultStroke);
export const Plus = makeIcon(PiPlusDefaultStroke);
export const PlusCircleIcon = makeIcon(PiPlusCircleStroke);
export const PlusIcon = makeIcon(PiPlusDefaultStroke);
export const PowerIcon = makeIcon(PiPowerDefaultStroke);
export const ProhibitIcon = makeIcon(PiBanRightStroke);
export const Pulse = makeIcon(PiActivityStroke);
export const PushPinIcon = makeIcon(PiPinDefaultStroke);
export const PushPinSlashIcon = makeIcon(PiPinSlantStroke);
export const QuestionIcon = makeIcon(PiQuestionMarkCircleStroke);
export const Receipt = makeIcon(PiReceipt01Stroke);
export const RobotIcon = makeIcon(PiBotStroke);
export const RocketLaunchIcon = makeIcon(PiRocketShipStroke);
export const SealCheckIcon = makeIcon(PiVerificationCheckStroke);
export const ShareFatIcon = makeIcon(PiShare01Stroke);
export const ShareIcon = makeIcon(PiShare01Stroke);
export const ShareNetworkIcon = makeIcon(PiShare02Stroke);
export const ShieldCheckIcon = makeIcon(PiShieldCheckStroke);
export const ShieldIcon = makeIcon(PiShieldStroke);
export const SidebarSimpleIcon = makeIcon(PiSidebarDefaultStroke);
export const SignInIcon = makeIcon(PiLogInRightStroke);
export const SignOut = makeIcon(PiLogOutRightStroke);
export const SignOutIcon = makeIcon(PiLogOutRightStroke);
export const SlidersHorizontalIcon = makeIcon(PiFilterHorizontalStroke);
export const SmileySadIcon = makeIcon(PiFaceSadStroke);
export const SparkleIcon = makeIcon(PiSparkleAI01Stroke);
export const SpeakerHigh = makeIcon(PiSpeakerOnStroke);
export const Spinner = makeIcon(PiSpinnerStroke);
export const SpinnerGapIcon = makeIcon(PiSpinnerStroke);
export const SpinnerIcon = makeIcon(PiSpinnerStroke);
export const SquareIcon = makeIcon(PiSquareStroke);
export const SquaresFour = makeIcon(PiGrid01Stroke);
export const SquaresFourIcon = makeIcon(PiGrid01Stroke);
export const StackIcon = makeIcon(PiLayersToStroke);
export const Star = makeIcon(PiStarStroke);
export const StarIcon = makeIcon(PiStarStroke);
export const Stop = makeIcon(PiStopBigStroke);
export const StopCircleIcon = makeIcon(PiStopCircleStroke);
export const StopIcon = makeIcon(PiStopSquareStroke);
export const StorefrontIcon = makeIcon(PiStoreDefaultStroke);
export const SunIcon = makeIcon(PiSunStroke);
export const Table = makeIcon(PiGridTableStroke);
export const TableIcon = makeIcon(PiGridTableStroke);
export const TerminalIcon = makeIcon(PiTerminalConsoleSquareStroke);
export const TextBIcon = makeIcon(PiBoldStroke);
export const TextItalicIcon = makeIcon(PiItalicStroke);
export const TextStrikethroughIcon = makeIcon(PiStrikeThroughStroke);
export const ThumbsDown = makeIcon(PiThumbReactionDislikeStroke);
export const ThumbsUp = makeIcon(PiThumbReactionLikeStroke);
export const TiktokLogo = makeIcon(PiTiktokStroke);
export const Trash = makeIcon(PiDeleteDustbin01Stroke);
export const TrashIcon = makeIcon(PiDeleteDustbin01Stroke);
export const Tray = makeIcon(PiInboxDefaultStroke);
export const TreeStructureIcon = makeIcon(PiGitBranchStroke);
export const TwitterLogoIcon = makeIcon(PiTwitterStroke);
export const UploadIcon = makeIcon(PiUploadUpStroke);
export const UploadSimple = makeIcon(PiUploadUpStroke);
export const UploadSimpleIcon = makeIcon(PiUploadUpStroke);
export const User = makeIcon(PiUserDefaultStroke);
export const UserCheck = makeIcon(PiUserCheckStroke);
export const UserCircle = makeIcon(PiUserCircleStroke);
export const UserCircleDashedIcon = makeIcon(PiUserCircleDottedStroke);
export const UserCircleIcon = makeIcon(PiUserCircleStroke);
export const UserIcon = makeIcon(PiUserDefaultStroke);
export const UserMinus = makeIcon(PiUserRemoveStroke);
export const UserPlus = makeIcon(PiUserPlusStroke);
export const Users = makeIcon(PiUserTwoStroke);
export const UsersIcon = makeIcon(PiUserTwoStroke);
export const VideoCamera = makeIcon(PiVideoRecordingStroke);
export const VideoCameraIcon = makeIcon(PiVideoRecordingStroke);
export const WalletIcon = makeIcon(PiWalletDefaultStroke);
export const Warning = makeIcon(PiAlertTriangleStroke);
export const WarningCircle = makeIcon(PiAlertCircleStroke);
export const WarningCircleIcon = makeIcon(PiAlertCircleStroke);
export const WarningDiamondIcon = makeIcon(PiAlertTriangleStroke);
export const WarningIcon = makeIcon(PiAlertTriangleStroke);
export const WarningOctagonIcon = makeIcon(PiAlertTriangleStroke);
export const WebhooksLogoIcon = makeIcon(PiWebhookStroke);
export const X = makeIcon(PiMultipleCrossCancelDefaultStroke);
export const XCircleIcon = makeIcon(PiMultipleCrossCancelCircleStroke);
export const XIcon = makeIcon(PiMultipleCrossCancelDefaultStroke);
export const XLogo = makeIcon(PiXComStroke);
export const YoutubeLogo = makeIcon(PiYoutubeStroke);
