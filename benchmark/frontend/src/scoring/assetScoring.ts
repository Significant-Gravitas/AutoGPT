// Per-asset scoring module for the Signal Engine dashboard.
// Builds a 7-factor base score then applies asset-specific overlays.

export interface ScoreInput {
  fg_value: number;
  btc_chg: number | null;
  eth_chg: number | null;
  btc_dominance: number | null;
  funding_rate: number | null;
  dxy_change: number | null;
  oil_brent: number | null;
}

export type AssetBias = "CALL" | "PUT" | "WAIT";
export type Confidence = "high" | "medium" | "low";
export type AssetCategory = "crypto" | "forex" | "commodity" | "stock" | "index";

export interface AssetResult {
  score: number;
  bias: AssetBias;
  expiry: number;
  confidence: Confidence;
  baseScore: number;
  overlayAdjustment: number;
}

type OverlayFn = (d: ScoreInput, params: Record<string, number>) => number;

interface AssetConfig {
  category: AssetCategory;
  expiry: number;
  overlay: OverlayFn;
  params?: Record<string, number>;
}

// ── Base 7-factor engine ──────────────────────────────────────────────────────

function calcBaseScore(d: ScoreInput): { base: number; factorCount: number } {
  let s = 0;
  let fc = 0;

  const btc = d.btc_chg ?? 0;
  const eth = d.eth_chg ?? 0;
  const dom = d.btc_dominance ?? 52;

  // F1: Fear & Greed (contrarian)
  if (d.fg_value <= 25)      { s += 2; fc++; }
  else if (d.fg_value <= 45) { s += 1; fc++; }
  else if (d.fg_value >= 75) { s -= 1; fc++; }

  // F2: BTC 24h momentum
  if (btc > 4)       { s += 2; fc++; }
  else if (btc > 1)  { s += 1; fc++; }
  else if (btc < -4) { s -= 2; fc++; }
  else if (btc < -1) { s -= 1; fc++; }

  // F3: ETH 24h momentum (alt proxy)
  if (eth > 3)       { s += 1; fc++; }
  else if (eth < -3) { s -= 1; fc++; }

  // F4: BTC Dominance
  if (dom > 58)      { s -= 1; fc++; }
  else if (dom < 45) { s += 1; fc++; }

  // F5: Funding rate
  if (d.funding_rate !== null) {
    if (d.funding_rate < -0.01)  { s += 2; fc++; }
    else if (d.funding_rate < 0) { s += 1; fc++; }
    else if (d.funding_rate > 0.10) { s -= 1; fc++; }
  }

  // F6: DXY direction (falling DXY → risk-on)
  if (d.dxy_change !== null) {
    if (d.dxy_change < -0.5)      { s += 1; fc++; }
    else if (d.dxy_change > 0.5)  { s -= 1; fc++; }
  }

  // F7: BTC + ETH aligned
  const btcUp = btc > 0;
  const ethUp = eth > 0;
  if (Math.abs(btc) > 1 && ((btcUp && ethUp) || (!btcUp && !ethUp))) {
    s += 1; fc++;
  }

  return { base: Math.max(0, Math.min(7, s)), factorCount: fc };
}

// ── Overlay functions ─────────────────────────────────────────────────────────

function forexMajorOverlay(d: ScoreInput, params: Record<string, number>): number {
  let adj = 0;
  const w = params.dxyFactor ?? 1;
  if (d.dxy_change !== null) {
    if (d.dxy_change < -0.3) adj += w;
    else if (d.dxy_change > 0.3) adj -= w;
  }
  if (d.fg_value <= 20) adj += 1;
  return adj;
}

function commodityFXOverlay(d: ScoreInput): number {
  let adj = 0;
  if (d.oil_brent !== null && d.oil_brent > 100) adj += 1;
  if (d.fg_value <= 25) adj += 1;
  return adj;
}

function oilFXOverlay(d: ScoreInput): number {
  // Strong oil → CAD/NOK strength → bearish USD/CAD
  return d.oil_brent !== null && d.oil_brent > 100 ? -1 : 0;
}

function crossFXOverlay(d: ScoreInput): number {
  if (d.fg_value <= 25) return 1;
  if (d.fg_value >= 75) return -1;
  return 0;
}

function cryptoOverlay(d: ScoreInput): number {
  let adj = 0;
  if (d.funding_rate !== null && d.funding_rate < -0.02) adj += 1;
  if (d.dxy_change !== null && d.dxy_change < -0.5) adj += 1;
  if ((d.btc_dominance ?? 52) > 60) adj -= 1;
  return adj;
}

function goldOverlay(d: ScoreInput): number {
  let adj = 0;
  if (d.dxy_change !== null) {
    // DXY inverse is the dominant driver for gold — weight 2×
    if (d.dxy_change < -0.3) adj += 2;
    else if (d.dxy_change > 0.3) adj -= 2;
  }
  if (d.oil_brent !== null && d.oil_brent > 100) adj += 1;
  if (d.fg_value <= 25) adj += 1;
  return adj;
}

function silverOverlay(d: ScoreInput): number {
  let adj = 0;
  if ((d.btc_chg ?? 0) > 2) adj += 1;
  if (d.dxy_change !== null && d.dxy_change < -0.3) adj += 1;
  return adj;
}

function oilCommodityOverlay(d: ScoreInput): number {
  return d.dxy_change !== null && d.dxy_change < -0.3 ? 1 : 0;
}

function stockOverlay(d: ScoreInput): number {
  let adj = 0;
  if ((d.btc_chg ?? 0) > 3 && (d.eth_chg ?? 0) > 2) adj += 1;
  if (d.fg_value >= 60) adj += 1;
  return adj;
}

function indexOverlay(d: ScoreInput): number {
  return d.dxy_change !== null && d.dxy_change < -0.3 ? 1 : 0;
}

// ── Asset config map ──────────────────────────────────────────────────────────

export const ASSET_CONFIG: Record<string, AssetConfig> = {
  // Forex Majors
  "EUR/USD": { category: "forex",     expiry: 5,  overlay: forexMajorOverlay, params: { dxyFactor: 2 } },
  "USD/JPY": { category: "forex",     expiry: 5,  overlay: forexMajorOverlay, params: { dxyFactor: 2 } },
  "GBP/USD": { category: "forex",     expiry: 5,  overlay: forexMajorOverlay, params: { dxyFactor: 1 } },
  "USD/CHF": { category: "forex",     expiry: 5,  overlay: forexMajorOverlay, params: { dxyFactor: 1 } },
  "AUD/USD": { category: "forex",     expiry: 5,  overlay: commodityFXOverlay },
  "NZD/USD": { category: "forex",     expiry: 5,  overlay: commodityFXOverlay },
  "USD/CAD": { category: "forex",     expiry: 5,  overlay: oilFXOverlay },
  // Forex Crosses
  "EUR/GBP": { category: "forex",     expiry: 5,  overlay: crossFXOverlay },
  "EUR/JPY": { category: "forex",     expiry: 5,  overlay: crossFXOverlay },
  "EUR/CHF": { category: "forex",     expiry: 10, overlay: crossFXOverlay },
  "GBP/JPY": { category: "forex",     expiry: 5,  overlay: crossFXOverlay },
  "GBP/CHF": { category: "forex",     expiry: 10, overlay: crossFXOverlay },
  "AUD/JPY": { category: "forex",     expiry: 5,  overlay: commodityFXOverlay },
  "NZD/JPY": { category: "forex",     expiry: 5,  overlay: crossFXOverlay },
  "CAD/JPY": { category: "forex",     expiry: 5,  overlay: crossFXOverlay },
  // Crypto
  "BTC/USD":  { category: "crypto",   expiry: 15, overlay: cryptoOverlay },
  "ETH/USD":  { category: "crypto",   expiry: 15, overlay: cryptoOverlay },
  "SOL/USD":  { category: "crypto",   expiry: 15, overlay: cryptoOverlay },
  "XRP/USD":  { category: "crypto",   expiry: 15, overlay: cryptoOverlay },
  "BNB/USD":  { category: "crypto",   expiry: 15, overlay: cryptoOverlay },
  "LTC/USD":  { category: "crypto",   expiry: 15, overlay: cryptoOverlay },
  "BCH/USD":  { category: "crypto",   expiry: 15, overlay: cryptoOverlay },
  "EOS/USD":  { category: "crypto",   expiry: 15, overlay: cryptoOverlay },
  "DOGE/USD": { category: "crypto",   expiry: 15, overlay: cryptoOverlay },
  "ADA/USD":  { category: "crypto",   expiry: 15, overlay: cryptoOverlay },
  "DOT/USD":  { category: "crypto",   expiry: 15, overlay: cryptoOverlay },
  "AVAX/USD": { category: "crypto",   expiry: 15, overlay: cryptoOverlay },
  "LINK/USD": { category: "crypto",   expiry: 15, overlay: cryptoOverlay },
  // Commodities
  "Gold":      { category: "commodity", expiry: 5,  overlay: goldOverlay },
  "Silver":    { category: "commodity", expiry: 5,  overlay: silverOverlay },
  "Brent":     { category: "commodity", expiry: 5,  overlay: oilCommodityOverlay },
  "WTI":       { category: "commodity", expiry: 5,  overlay: oilCommodityOverlay },
  "Nat Gas":   { category: "commodity", expiry: 10, overlay: oilCommodityOverlay },
  "Platinum":  { category: "commodity", expiry: 10, overlay: silverOverlay },
  "Palladium": { category: "commodity", expiry: 10, overlay: silverOverlay },
  // Stocks
  "AAPL":  { category: "stock", expiry: 15, overlay: stockOverlay },
  "TSLA":  { category: "stock", expiry: 15, overlay: stockOverlay },
  "AMZN":  { category: "stock", expiry: 15, overlay: stockOverlay },
  "MSFT":  { category: "stock", expiry: 15, overlay: stockOverlay },
  "NVDA":  { category: "stock", expiry: 15, overlay: stockOverlay },
  "META":  { category: "stock", expiry: 15, overlay: stockOverlay },
  "NFLX":  { category: "stock", expiry: 15, overlay: stockOverlay },
  "JPM":   { category: "stock", expiry: 15, overlay: stockOverlay },
  "GOOGL": { category: "stock", expiry: 15, overlay: stockOverlay },
  "AMD":   { category: "stock", expiry: 15, overlay: stockOverlay },
  "V":     { category: "stock", expiry: 15, overlay: stockOverlay },
  "DIS":   { category: "stock", expiry: 15, overlay: stockOverlay },
  // Indices
  "US30":  { category: "index", expiry: 5,  overlay: indexOverlay },
  "US500": { category: "index", expiry: 5,  overlay: indexOverlay },
  "US100": { category: "index", expiry: 5,  overlay: indexOverlay },
  "JP225": { category: "index", expiry: 5,  overlay: indexOverlay },
  "UK100": { category: "index", expiry: 10, overlay: indexOverlay },
  "DE40":  { category: "index", expiry: 10, overlay: indexOverlay },
  "FR40":  { category: "index", expiry: 10, overlay: indexOverlay },
  "AU200": { category: "index", expiry: 10, overlay: indexOverlay },
  "NL25":  { category: "index", expiry: 10, overlay: indexOverlay },
};

// ── Public API ────────────────────────────────────────────────────────────────

export function calcAssetScore(assetKey: string, d: ScoreInput): AssetResult {
  const config = ASSET_CONFIG[assetKey];
  const { base, factorCount } = calcBaseScore(d);

  if (!config) {
    return { score: base, bias: biasLabel(base), expiry: 5, confidence: "low", baseScore: base, overlayAdjustment: 0 };
  }

  const adj = config.overlay(d, config.params ?? {});
  const finalScore = Math.max(0, Math.min(7, Math.round(base + adj)));
  const confidence: Confidence = factorCount >= 4 ? "high" : factorCount >= 2 ? "medium" : "low";

  return { score: finalScore, bias: biasLabel(finalScore), expiry: config.expiry, confidence, baseScore: base, overlayAdjustment: adj };
}

function biasLabel(score: number): AssetBias {
  if (score >= 5) return "CALL";
  if (score <= 2) return "PUT";
  return "WAIT";
}
