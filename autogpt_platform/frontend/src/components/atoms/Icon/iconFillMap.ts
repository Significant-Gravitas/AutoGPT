// Solid (filled) counterparts for the AutoGPT stroke icons, keyed by the
// stroke export name used in iconMap/registry. createIcon consults this when a
// callsite passes weight="fill", so filled-vs-regular state toggles (favorite
// hearts, vote thumbs, connected node handles) stay visually distinct when the
// AutoGPT set is active. A stroke icon without an entry here simply renders
// its stroke variant for every weight. Values must exist in
// `@autogpt/icons/solid` and be imported in agptIcons.js.
export const iconFillMap: Record<string, string> = {
  AlertCircleStroke: "AlertCircleSolid",
  AlertTriangleStroke: "AlertTriangleSolid",
  BulbBoltStroke: "BulbBoltSolid",
  BulbDefaultStroke: "BulbDefaultSolid",
  ChatDefaultStroke: "ChatDefaultSolid",
  CheckTickCircleStroke: "CheckTickCircleSolid",
  CheckTickSquareStroke: "CheckTickSquareSolid",
  // The set has no plain solid circle; the dot variant still reads as a
  // filled/connected state next to the hollow CircleStroke.
  CircleStroke: "CircleDotSolid",
  ClockDefaultStroke: "ClockDefaultSolid",
  DiscordStroke: "DiscordSolid",
  FilterFunnelStroke: "FilterFunnelSolid",
  FolderDefaultStroke: "FolderDefaultSolid",
  HeartStroke: "HeartSolid",
  LabFlaskConicalStroke: "LabFlaskConicalSolid",
  MultipleCrossCancelCircleStroke: "MultipleCrossCancelCircleSolid",
  NotificationBellOnStroke: "NotificationBellOnSolid",
  PauseCircleStroke: "PauseCircleSolid",
  PinDefaultStroke: "PinDefaultSolid",
  PlayBigStroke: "PlayBigSolid",
  PlayCircleStroke: "PlayCircleSolid",
  PlusCircleStroke: "PlusCircleSolid",
  ShieldCheckStroke: "ShieldCheckSolid",
  SparkleAI01Stroke: "SparkleAI01Solid",
  StarStroke: "StarSolid",
  StopBigStroke: "StopBigSolid",
  ThumbReactionDislikeStroke: "ThumbReactionDislikeSolid",
  ThumbReactionLikeStroke: "ThumbReactionLikeSolid",
  UserDefaultStroke: "UserDefaultSolid",
};
