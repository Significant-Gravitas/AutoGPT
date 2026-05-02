import { useState, useEffect, useCallback } from "react";
import Head from "next/head";
import { calcAssetScore, type ScoreInput, type AssetBias } from "../scoring/assetScoring";

/* ── Palette ─────────────────────────────────────────────────────────────── */
const C = {
  bg: "#060a0e", card: "#0d1117", border: "#1e2d3d",
  green: "#00ff88", red: "#ff4560", yellow: "#f0c040",
  blue: "#38bdf8", text: "#e2e8f0", dim: "#64748b",
};

/* ── API ──────────────────────────────────────────────────────────────────── */
const CG_IDS = "bitcoin,ethereum,solana,ripple,binancecoin,litecoin,bitcoin-cash,eos,dogecoin,cardano,polkadot,avalanche-2,chainlink";
const CG_PRICE  = `https://api.coingecko.com/api/v3/simple/price?ids=${CG_IDS}&vs_currencies=usd&include_24hr_change=true`;
const CG_GLOBAL = "https://api.coingecko.com/api/v3/global";
const FG_URL    = "https://api.alternative.me/fng/?limit=1";

/* ── Static asset metadata ───────────────────────────────────────────────── */
const CRYPTO_TOKENS = [
  { key: "BTC/USD",  cg: "bitcoin",      prk: "btc",  symbol: "BTC/USD",  role: "Primary macro asset" },
  { key: "ETH/USD",  cg: "ethereum",     prk: "eth",  symbol: "ETH/USD",  role: "Alt market proxy" },
  { key: "SOL/USD",  cg: "solana",       prk: "sol",  symbol: "SOL/USD",  role: "High-beta risk signal" },
  { key: "XRP/USD",  cg: "ripple",       prk: "xrp",  symbol: "XRP/USD",  role: "Liquidity indicator" },
  { key: "BNB/USD",  cg: "binancecoin",  prk: "bnb",  symbol: "BNB/USD",  role: "Exchange flow signal" },
  { key: "LTC/USD",  cg: "litecoin",     prk: "ltc",  symbol: "LTC/USD",  role: "BTC silver proxy" },
  { key: "BCH/USD",  cg: "bitcoin-cash", prk: "bch",  symbol: "BCH/USD",  role: "Fork / payment narrative" },
  { key: "EOS/USD",  cg: "eos",          prk: "eos",  symbol: "EOS/USD",  role: "Legacy platform coin" },
  { key: "DOGE/USD", cg: "dogecoin",     prk: "doge", symbol: "DOGE/USD", role: "Meme / social sentiment" },
  { key: "ADA/USD",  cg: "cardano",      prk: "ada",  symbol: "ADA/USD",  role: "Smart contract platform" },
  { key: "DOT/USD",  cg: "polkadot",     prk: "dot",  symbol: "DOT/USD",  role: "Interoperability layer" },
  { key: "AVAX/USD", cg: "avalanche-2",  prk: "avax", symbol: "AVAX/USD", role: "L1 competition / high-beta" },
  { key: "LINK/USD", cg: "chainlink",    prk: "link", symbol: "LINK/USD", role: "Oracle / DeFi proxy" },
];

const FOREX_TOKENS = [
  { key: "EUR/USD", symbol: "EUR/USD", role: "Euro vs dollar — DXY inverse" },
  { key: "USD/JPY", symbol: "USD/JPY", role: "Dollar vs yen — safe haven" },
  { key: "GBP/USD", symbol: "GBP/USD", role: "Cable — UK macro divergence" },
  { key: "USD/CHF", symbol: "USD/CHF", role: "Dollar vs franc — risk gauge" },
  { key: "AUD/USD", symbol: "AUD/USD", role: "Commodity FX — risk proxy" },
  { key: "NZD/USD", symbol: "NZD/USD", role: "Commodity FX — risk proxy" },
  { key: "USD/CAD", symbol: "USD/CAD", role: "Oil FX — energy correlation" },
  { key: "EUR/GBP", symbol: "EUR/GBP", role: "EU vs UK macro divergence" },
  { key: "EUR/JPY", symbol: "EUR/JPY", role: "Risk barometer — equity proxy" },
  { key: "EUR/CHF", symbol: "EUR/CHF", role: "Low vol — SNB floor watch" },
  { key: "GBP/JPY", symbol: "GBP/JPY", role: "Volatility king — high beta" },
  { key: "GBP/CHF", symbol: "GBP/CHF", role: "UK safe-haven flows" },
  { key: "AUD/JPY", symbol: "AUD/JPY", role: "Commodity + risk barometer" },
  { key: "NZD/JPY", symbol: "NZD/JPY", role: "OTC commodity risk signal" },
  { key: "CAD/JPY", symbol: "CAD/JPY", role: "Oil + risk sentiment" },
];

const COMMODITY_TOKENS = [
  { key: "Gold",      symbol: "XAU/USD", role: "Primary safe haven — DXY inverse" },
  { key: "Silver",    symbol: "XAG/USD", role: "Industrial + precious dual role" },
  { key: "Brent",     symbol: "UKOIL",   role: "Geopolitical risk premium" },
  { key: "WTI",       symbol: "USOIL",   role: "US crude — EIA inventory driven" },
  { key: "Nat Gas",   symbol: "NATGAS",  role: "Weather / storage seasonal" },
  { key: "Platinum",  symbol: "XPT/USD", role: "Auto sector proxy" },
  { key: "Palladium", symbol: "XPD/USD", role: "Russian supply risk" },
];

const STOCK_TOKENS = [
  { key: "AAPL",  symbol: "AAPL",  role: "Tech bellwether — earnings driven" },
  { key: "TSLA",  symbol: "TSLA",  role: "High-beta EV — follows crypto risk" },
  { key: "AMZN",  symbol: "AMZN",  role: "Consumer + cloud proxy" },
  { key: "MSFT",  symbol: "MSFT",  role: "Enterprise tech / AI narrative" },
  { key: "NVDA",  symbol: "NVDA",  role: "AI chip bellwether — highest beta" },
  { key: "META",  symbol: "META",  role: "Social media + AI ad proxy" },
  { key: "NFLX",  symbol: "NFLX",  role: "Consumer discretionary" },
  { key: "JPM",   symbol: "JPM",   role: "Financials — rate & NFP sensitive" },
  { key: "GOOGL", symbol: "GOOGL", role: "Search + cloud + AI" },
  { key: "AMD",   symbol: "AMD",   role: "Chip sector — NVDA beta" },
  { key: "V",     symbol: "V",     role: "Consumer spending flow" },
  { key: "DIS",   symbol: "DIS",   role: "Consumer discretionary" },
];

const INDEX_TOKENS = [
  { key: "US500", symbol: "US500 (S&P)", role: "Broadest macro barometer" },
  { key: "US100", symbol: "US100 (NDQ)", role: "Tech-heavy — rate sensitive" },
  { key: "US30",  symbol: "US30 (DOW)",  role: "Blue-chip industrial proxy" },
  { key: "JP225", symbol: "JP225 (NKY)", role: "Yen-sensitive — Asian session" },
  { key: "UK100", symbol: "UK100 (FTS)", role: "Commodity/energy heavy" },
  { key: "DE40",  symbol: "DE40 (DAX)",  role: "EU manufacturing — China proxy" },
  { key: "FR40",  symbol: "FR40 (CAC)",  role: "Luxury sector heavy" },
  { key: "AU200", symbol: "AU200",        role: "Mining / China demand proxy" },
  { key: "NL25",  symbol: "NL25 (AEX)",  role: "Semiconductor sector — ASML" },
];

type TabKey = "crypto" | "forex" | "commodity" | "stock" | "index";

const TABS: { key: TabKey; label: string }[] = [
  { key: "crypto",    label: "CRYPTO" },
  { key: "forex",     label: "FOREX" },
  { key: "commodity", label: "METALS & OIL" },
  { key: "stock",     label: "STOCKS" },
  { key: "index",     label: "INDICES" },
];

/* ── Market data shape ───────────────────────────────────────────────────── */
interface MarketData {
  btc_price: number | null; btc_chg: number | null;
  eth_price: number | null; eth_chg: number | null;
  sol_price: number | null; sol_chg: number | null;
  xrp_price: number | null; xrp_chg: number | null;
  bnb_price: number | null; bnb_chg: number | null;
  ltc_price: number | null; ltc_chg: number | null;
  bch_price: number | null; bch_chg: number | null;
  eos_price: number | null; eos_chg: number | null;
  doge_price: number | null; doge_chg: number | null;
  ada_price: number | null; ada_chg: number | null;
  dot_price: number | null; dot_chg: number | null;
  avax_price: number | null; avax_chg: number | null;
  link_price: number | null; link_chg: number | null;
  btc_dominance: number | null;
  total_mcap: string;
  fg_value: number;
  fg_label: string;
}

/* ── Helpers ─────────────────────────────────────────────────────────────── */
function fmtPrice(p: number | null, key: string): string {
  if (p == null) return "—";
  if (key === "xrp" || key === "doge" || key === "ada" || key === "eos") return `$${p.toFixed(4)}`;
  if (p < 10) return `$${p.toFixed(3)}`;
  if (p < 200) return `$${p.toFixed(2)}`;
  return `$${p.toLocaleString("en-US", { maximumFractionDigits: 2 })}`;
}

function toScoreInput(m: MarketData): ScoreInput {
  return {
    fg_value: m.fg_value,
    btc_chg: m.btc_chg,
    eth_chg: m.eth_chg,
    btc_dominance: m.btc_dominance,
    funding_rate: null,
    dxy_change: null,
    oil_brent: null,
  };
}

function biasColor(b: AssetBias | string): string {
  if (b === "CALL") return C.green;
  if (b === "PUT")  return C.red;
  return C.yellow;
}

function scoreBarColor(s: number): string {
  if (s >= 5) return C.green;
  if (s <= 2) return C.red;
  return C.yellow;
}

/* ── Spinner ─────────────────────────────────────────────────────────────── */
function Spin({ size = 20, color = C.green }: { size?: number; color?: string }) {
  return (
    <div style={{
      width: size, height: size, border: `2px solid #1e2d3d`,
      borderTop: `2px solid ${color}`, borderRadius: "50%",
      animation: "spin 0.7s linear infinite", flexShrink: 0,
    }} />
  );
}

/* ── Score bar ───────────────────────────────────────────────────────────── */
function ScoreBar({ score }: { score: number }) {
  const col = scoreBarColor(score);
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
      <div style={{ background: "#0a1018", borderRadius: 4, height: 5, width: 52, overflow: "hidden", flexShrink: 0 }}>
        <div style={{ width: `${(score / 7) * 100}%`, height: "100%", background: col, borderRadius: 4, transition: "width 0.6s ease" }} />
      </div>
      <span style={{ color: col, fontSize: 9, fontWeight: 800, fontFamily: "monospace" }}>{score}/7</span>
    </div>
  );
}

/* ── Signal badge ────────────────────────────────────────────────────────── */
function SignalBadge({ bias }: { bias: AssetBias }) {
  const col = biasColor(bias);
  return (
    <div style={{ background: `${col}20`, border: `1px solid ${col}55`, borderRadius: 7, padding: "6px 0", textAlign: "center", minWidth: 62 }}>
      <span style={{ color: col, fontSize: 11, fontWeight: 900 }}>{bias}</span>
    </div>
  );
}

/* ── Main component ──────────────────────────────────────────────────────── */
export default function SignalEngine() {
  const [market, setMarket]       = useState<MarketData | null>(null);
  const [loading, setLoading]     = useState(true);
  const [error, setError]         = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<TabKey>("crypto");

  const fetchMarketData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [priceRes, globalRes, fgRes] = await Promise.all([
        fetch(CG_PRICE), fetch(CG_GLOBAL), fetch(FG_URL),
      ]);
      if (!priceRes.ok || !globalRes.ok || !fgRes.ok) throw new Error("API error");

      const pd = await priceRes.json() as Record<string, Record<string, number>>;
      const gd = await globalRes.json() as { data: { market_cap_percentage: Record<string, number>; total_market_cap: Record<string, number> } };
      const fd = await fgRes.json() as { data: Array<{ value: string; value_classification: string }> };

      const p = (id: string, k: string) => pd[id]?.[k] ?? null;

      setMarket({
        btc_price:  p("bitcoin",      "usd"),  btc_chg:  p("bitcoin",      "usd_24h_change"),
        eth_price:  p("ethereum",     "usd"),  eth_chg:  p("ethereum",     "usd_24h_change"),
        sol_price:  p("solana",       "usd"),  sol_chg:  p("solana",       "usd_24h_change"),
        xrp_price:  p("ripple",       "usd"),  xrp_chg:  p("ripple",       "usd_24h_change"),
        bnb_price:  p("binancecoin",  "usd"),  bnb_chg:  p("binancecoin",  "usd_24h_change"),
        ltc_price:  p("litecoin",     "usd"),  ltc_chg:  p("litecoin",     "usd_24h_change"),
        bch_price:  p("bitcoin-cash", "usd"),  bch_chg:  p("bitcoin-cash", "usd_24h_change"),
        eos_price:  p("eos",          "usd"),  eos_chg:  p("eos",          "usd_24h_change"),
        doge_price: p("dogecoin",     "usd"),  doge_chg: p("dogecoin",     "usd_24h_change"),
        ada_price:  p("cardano",      "usd"),  ada_chg:  p("cardano",      "usd_24h_change"),
        dot_price:  p("polkadot",     "usd"),  dot_chg:  p("polkadot",     "usd_24h_change"),
        avax_price: p("avalanche-2",  "usd"),  avax_chg: p("avalanche-2",  "usd_24h_change"),
        link_price: p("chainlink",    "usd"),  link_chg: p("chainlink",    "usd_24h_change"),
        btc_dominance: gd.data?.market_cap_percentage?.btc ?? null,
        total_mcap: ((gd.data?.total_market_cap?.usd ?? 0) / 1e12).toFixed(2),
        fg_value: parseInt(fd.data?.[0]?.value ?? "50", 10),
        fg_label: fd.data?.[0]?.value_classification ?? "—",
      });
      setLastUpdate(new Date().toLocaleTimeString());
    } catch {
      setError("Failed to fetch live data. Check your connection.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void fetchMarketData();
    const iv = setInterval(() => void fetchMarketData(), 90_000);
    return () => clearInterval(iv);
  }, [fetchMarketData]);

  /* ── Derive composite signal & factors ─────────────────────────────────── */
  const si: ScoreInput | null = market ? toScoreInput(market) : null;
  const overall = si ? calcAssetScore("BTC/USD", si) : null;
  const score = overall?.score ?? 0;
  const bias  = overall?.bias  ?? "WAIT";
  const scoreColor = scoreBarColor(score);
  const bColor = biasColor(bias);

  const factors = market
    ? [
        {
          label: "Fear & Greed",
          value: `${market.fg_value} — ${market.fg_label}`,
          detail: market.fg_value <= 25 ? "Extreme fear — contrarian buy zone"
                : market.fg_value <= 45 ? "Fear zone — possible accumulation"
                : market.fg_value >= 75 ? "Greed — reversal risk elevated"
                : "Neutral — awaiting catalyst",
          status: (market.fg_value <= 45 ? "B" : market.fg_value >= 75 ? "R" : "N") as "B" | "R" | "N",
        },
        {
          label: "BTC 24h Momentum",
          value: market.btc_chg != null ? `${market.btc_chg > 0 ? "+" : ""}${market.btc_chg.toFixed(2)}%` : "—",
          detail: (market.btc_chg ?? 0) > 1 ? "Positive — trend following"
                : (market.btc_chg ?? 0) < -1 ? "Negative — caution"
                : "Sideways — low conviction",
          status: ((market.btc_chg ?? 0) > 1 ? "B" : (market.btc_chg ?? 0) < -1 ? "R" : "N") as "B" | "R" | "N",
        },
        {
          label: "ETH / Alt Proxy",
          value: market.eth_chg != null ? `${market.eth_chg > 0 ? "+" : ""}${market.eth_chg.toFixed(2)}%` : "—",
          detail: (market.eth_chg ?? 0) > 3 ? "Alts strong — broad participation"
                : (market.eth_chg ?? 0) < -3 ? "Alts weak — risk-off"
                : "Mild alt action",
          status: ((market.eth_chg ?? 0) > 3 ? "B" : (market.eth_chg ?? 0) < -3 ? "R" : "N") as "B" | "R" | "N",
        },
        {
          label: "BTC Dominance",
          value: market.btc_dominance != null ? `${market.btc_dominance.toFixed(1)}%` : "—",
          detail: (market.btc_dominance ?? 52) > 58 ? "Bitcoin Season — capital in BTC"
                : (market.btc_dominance ?? 52) < 45 ? "Alt Season — capital rotating"
                : "Balanced — watch for rotation",
          status: ((market.btc_dominance ?? 52) > 58 ? "R" : (market.btc_dominance ?? 52) < 45 ? "B" : "N") as "B" | "R" | "N",
        },
        {
          label: "Cross-Asset Confirm",
          value: ((market.btc_chg ?? 0) > 0 && (market.eth_chg ?? 0) > 0) ||
                 ((market.btc_chg ?? 0) < 0 && (market.eth_chg ?? 0) < 0) ? "ALIGNED" : "DIVERGENT",
          detail: "BTC & ETH moving together → conviction",
          status: (((market.btc_chg ?? 0) > 0 && (market.eth_chg ?? 0) > 0) ||
                   ((market.btc_chg ?? 0) < 0 && (market.eth_chg ?? 0) < 0) ? "B" : "N") as "B" | "R" | "N",
        },
        {
          label: "DXY Direction",
          value: "N/A — no live feed",
          detail: "Connect a forex API to enable DXY overlay scoring",
          status: "N" as "N",
        },
        {
          label: "Funding Rate",
          value: "N/A — no live feed",
          detail: "Connect an exchange API for perpetual funding data",
          status: "N" as "N",
        },
      ]
    : [];

  return (
    <>
      <Head>
        <title>Signal Engine · All Assets</title>
        <meta name="description" content="Live crypto signal engine — Pocket Option all-asset coverage" />
      </Head>
      <div style={{ background: C.bg, minHeight: "100vh", color: C.text, fontFamily: "'Courier New', monospace", paddingBottom: 52 }}>
        <style>{`
          @keyframes spin { to { transform: rotate(360deg); } }
          * { box-sizing: border-box; margin: 0; padding: 0; }
          ::-webkit-scrollbar { height: 3px; }
          ::-webkit-scrollbar-thumb { background: #1e2d3d; }
        `}</style>

        {/* ── Header ─────────────────────────────────────────────────────── */}
        <div style={{ background: "#0a0f1a", borderBottom: `1px solid ${C.border}`, padding: "14px 16px", position: "sticky", top: 0, zIndex: 10 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <div>
              <div style={{ color: C.green, fontSize: 15, fontWeight: 900, letterSpacing: "0.05em" }}>⬡ SIGNAL ENGINE</div>
              <div style={{ color: C.dim, fontSize: 9, letterSpacing: "0.14em", marginTop: 2 }}>100+ ASSETS · POCKET OPTION · LIVE API</div>
            </div>
            <button
              onClick={() => void fetchMarketData()}
              disabled={loading}
              style={{
                background: `${C.green}15`, border: `1px solid ${loading ? C.dim : C.green}`,
                color: loading ? C.dim : C.green, padding: "10px 18px", borderRadius: 8,
                fontSize: 12, fontWeight: 700, cursor: loading ? "not-allowed" : "pointer",
                fontFamily: "monospace", display: "flex", alignItems: "center", gap: 8,
              }}
            >
              {loading ? <><Spin size={14} />&nbsp;Refreshing</> : "↺ Refresh"}
            </button>
          </div>
          <div style={{ color: C.dim, fontSize: 9, marginTop: 5 }}>
            {lastUpdate ? `Last update: ${lastUpdate} · Auto-refresh 90s` : "Initializing live connection..."}
          </div>
        </div>

        <div style={{ padding: "14px", display: "flex", flexDirection: "column", gap: 14 }}>
          {/* ── Error ──────────────────────────────────────────────────── */}
          {error && (
            <div style={{ background: "#ff456015", border: `1px solid ${C.red}50`, borderRadius: 10, padding: 16, color: C.red, fontSize: 13, textAlign: "center" }}>
              {error}
              <button onClick={() => void fetchMarketData()} style={{ marginLeft: 10, background: "transparent", border: `1px solid ${C.red}`, color: C.red, borderRadius: 6, padding: "4px 12px", cursor: "pointer", fontSize: 12 }}>
                Retry
              </button>
            </div>
          )}

          {loading && !market && (
            <div style={{ textAlign: "center", padding: 40, color: C.dim }}>
              <Spin size={30} color={C.green} />
              <div style={{ marginTop: 16, fontSize: 13 }}>Loading live market data...</div>
            </div>
          )}

          {market && si && (
            <>
              {/* ── Overall Signal ──────────────────────────────────────── */}
              <div style={{ background: `${bColor}12`, border: `2px solid ${bColor}50`, borderRadius: 14, padding: "22px 16px", textAlign: "center", position: "relative", overflow: "hidden" }}>
                <div style={{ position: "absolute", inset: 0, background: `radial-gradient(ellipse at center,${bColor}10,transparent 70%)`, pointerEvents: "none" }} />
                <div style={{ color: C.dim, fontSize: 9, letterSpacing: "0.2em", marginBottom: 10 }}>POCKET OPTION · COMPOSITE SIGNAL</div>
                <div style={{ fontSize: 66, fontWeight: 900, color: bColor, lineHeight: 1, textShadow: `0 0 30px ${bColor}60` }}>{bias}</div>
                <div style={{ color: bColor, fontSize: 11, marginTop: 8, letterSpacing: "0.1em", opacity: 0.85 }}>
                  {bias === "CALL" ? "Conditions favor LONG entries" : bias === "PUT" ? "Conditions favor SHORT entries" : "Mixed signals — stand down"}
                </div>
                <div style={{ margin: "14px auto 0", maxWidth: 280 }}>
                  <div style={{ background: "#0a1018", borderRadius: 6, height: 8, overflow: "hidden" }}>
                    <div style={{ width: `${(score / 7) * 100}%`, height: "100%", background: scoreColor, boxShadow: `0 0 8px ${scoreColor}60`, borderRadius: 6, transition: "width 0.8s ease" }} />
                  </div>
                  <div style={{ display: "flex", justifyContent: "space-between", marginTop: 4 }}>
                    <span style={{ color: C.dim, fontSize: 9 }}>7-Factor Score</span>
                    <span style={{ color: scoreColor, fontSize: 10, fontWeight: 800 }}>{score} / 7</span>
                  </div>
                </div>
                <div style={{ display: "flex", gap: 8, justifyContent: "center", marginTop: 14, flexWrap: "wrap" }}>
                  {[
                    { label: "F&G",  val: `${market.fg_value} · ${market.fg_label}`, col: C.yellow },
                    { label: "Dom",  val: market.btc_dominance != null ? `${market.btc_dominance.toFixed(1)}%` : "—", col: C.yellow },
                    { label: "Mcap", val: `$${market.total_mcap}T`, col: C.blue },
                    { label: "DXY",  val: "N/A", col: C.dim },
                  ].map((p, i) => (
                    <div key={i} style={{ background: "#ffffff0a", border: `1px solid ${C.border}`, borderRadius: 20, padding: "5px 12px", fontSize: 10 }}>
                      <span style={{ color: C.dim }}>{p.label}: </span>
                      <b style={{ color: p.col }}>{p.val}</b>
                    </div>
                  ))}
                </div>
              </div>

              {/* ── Category Tabs ───────────────────────────────────────── */}
              <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 14, overflow: "hidden" }}>
                {/* Tab bar */}
                <div style={{ display: "flex", overflowX: "auto", borderBottom: `1px solid ${C.border}`, background: "#0a0f1a" }}>
                  {TABS.map(t => {
                    const active = activeTab === t.key;
                    return (
                      <button
                        key={t.key}
                        onClick={() => setActiveTab(t.key)}
                        style={{
                          flexShrink: 0, padding: "12px 16px", background: "transparent",
                          border: "none", borderBottom: active ? `2px solid ${C.green}` : "2px solid transparent",
                          color: active ? C.green : C.dim, fontSize: 10, fontWeight: 700,
                          letterSpacing: "0.1em", cursor: "pointer", fontFamily: "monospace",
                          transition: "color 0.2s",
                        }}
                      >
                        {t.label}
                      </button>
                    );
                  })}
                </div>

                {/* ── Crypto Tab ─────────────────────────────────────────── */}
                {activeTab === "crypto" && (
                  <>
                    <div style={{ display: "grid", gridTemplateColumns: "1.8fr 1.3fr 60px 70px 62px", padding: "8px 14px", borderBottom: `1px solid ${C.border}`, background: "#0a0f1a80" }}>
                      {["TOKEN", "PRICE", "24H", "SCORE", "SIG"].map(h => (
                        <div key={h} style={{ color: "#475569", fontSize: 9, letterSpacing: "0.12em", fontWeight: 700 }}>{h}</div>
                      ))}
                    </div>
                    {CRYPTO_TOKENS.map((t, i) => {
                      const price = market[`${t.prk}_price` as keyof MarketData] as number | null;
                      const chg   = market[`${t.prk}_chg`   as keyof MarketData] as number | null;
                      const res   = calcAssetScore(t.key, si);
                      const chgColor = (chg ?? 0) >= 0 ? C.green : C.red;
                      return (
                        <div key={t.key} style={{ display: "grid", gridTemplateColumns: "1.8fr 1.3fr 60px 70px 62px", padding: "13px 14px", alignItems: "center", borderBottom: i < CRYPTO_TOKENS.length - 1 ? `1px solid ${C.border}` : "none", background: `${biasColor(res.bias)}05` }}>
                          <div>
                            <div style={{ color: C.text, fontSize: 13, fontWeight: 800 }}>{t.symbol}</div>
                            <div style={{ color: "#475569", fontSize: 9, marginTop: 2 }}>{t.role}</div>
                          </div>
                          <div style={{ color: C.text, fontSize: 13, fontWeight: 800, fontFamily: "monospace" }}>
                            {fmtPrice(price, t.prk)}
                          </div>
                          <div>
                            <div style={{ color: chgColor, fontSize: 11, fontWeight: 700, fontFamily: "monospace" }}>
                              {chg != null ? `${chg > 0 ? "+" : ""}${chg.toFixed(2)}%` : "—"}
                            </div>
                            <div style={{ color: chgColor, fontSize: 9 }}>{(chg ?? 0) >= 0 ? "▲" : "▼"}</div>
                          </div>
                          <ScoreBar score={res.score} />
                          <SignalBadge bias={res.bias} />
                        </div>
                      );
                    })}
                  </>
                )}

                {/* ── Signal-only tabs (Forex / Commodity / Stock / Index) ── */}
                {activeTab !== "crypto" && (() => {
                  const tokens =
                    activeTab === "forex"     ? FOREX_TOKENS :
                    activeTab === "commodity" ? COMMODITY_TOKENS :
                    activeTab === "stock"     ? STOCK_TOKENS : INDEX_TOKENS;

                  return (
                    <>
                      <div style={{ background: "#0a1018", padding: "8px 14px", borderBottom: `1px solid ${C.border}` }}>
                        <span style={{ color: C.dim, fontSize: 9, letterSpacing: "0.1em" }}>
                          SIGNAL ONLY — no live price feed for this category · scores derived from macro data
                        </span>
                      </div>
                      <div style={{ display: "grid", gridTemplateColumns: "1fr 70px 62px 46px", padding: "8px 14px", borderBottom: `1px solid ${C.border}`, background: "#0a0f1a80" }}>
                        {["ASSET", "SCORE", "SIGNAL", "EXP"].map(h => (
                          <div key={h} style={{ color: "#475569", fontSize: 9, letterSpacing: "0.12em", fontWeight: 700 }}>{h}</div>
                        ))}
                      </div>
                      {tokens.map((t, i) => {
                        const res = calcAssetScore(t.key, si);
                        return (
                          <div key={t.key} style={{ display: "grid", gridTemplateColumns: "1fr 70px 62px 46px", padding: "13px 14px", alignItems: "center", borderBottom: i < tokens.length - 1 ? `1px solid ${C.border}` : "none", background: `${biasColor(res.bias)}05` }}>
                            <div>
                              <div style={{ color: C.text, fontSize: 13, fontWeight: 800 }}>{t.symbol}</div>
                              <div style={{ color: "#475569", fontSize: 9, marginTop: 2 }}>{t.role}</div>
                            </div>
                            <ScoreBar score={res.score} />
                            <SignalBadge bias={res.bias} />
                            <div style={{ color: C.dim, fontSize: 10, fontFamily: "monospace" }}>{res.expiry}m</div>
                          </div>
                        );
                      })}
                    </>
                  );
                })()}
              </div>

              {/* ── 7-Factor Breakdown ──────────────────────────────────── */}
              <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 14, overflow: "hidden" }}>
                <div style={{ background: "#0a0f1a", padding: "12px 16px", borderBottom: `1px solid ${C.border}` }}>
                  <div style={{ color: C.dim, fontSize: 10, letterSpacing: "0.14em", fontWeight: 700 }}>7-FACTOR SIGNAL BREAKDOWN</div>
                </div>
                {factors.map((f, i) => {
                  const col = f.status === "B" ? C.green : f.status === "R" ? C.red : C.yellow;
                  return (
                    <div key={i} style={{ display: "flex", alignItems: "flex-start", gap: 12, padding: "13px 16px", borderBottom: i < factors.length - 1 ? `1px solid ${C.border}` : "none", background: `${col}06` }}>
                      <span style={{ color: col, fontSize: 13, width: 14, flexShrink: 0, marginTop: 1, textAlign: "center" }}>
                        {f.status === "B" ? "▲" : f.status === "R" ? "▼" : "━"}
                      </span>
                      <div style={{ flex: 1 }}>
                        <div style={{ color: C.text, fontSize: 12, fontWeight: 700 }}>{f.label}</div>
                        <div style={{ color: C.dim, fontSize: 10, marginTop: 2 }}>{f.detail}</div>
                      </div>
                      <span style={{ color: col, fontSize: 11, fontFamily: "monospace", fontWeight: 700, flexShrink: 0, marginTop: 1 }}>{f.value}</span>
                    </div>
                  );
                })}
              </div>

              {/* ── Macro Regime Guide ───────────────────────────────────── */}
              <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 14, overflow: "hidden" }}>
                <div style={{ background: "#0a0f1a", padding: "12px 16px", borderBottom: `1px solid ${C.border}` }}>
                  <div style={{ color: C.dim, fontSize: 10, letterSpacing: "0.14em", fontWeight: 700 }}>MACRO REGIME GUIDE</div>
                </div>
                {[
                  { regime: "Hawkish Fed / Strong DXY", crypto: "PUT", forex: "CALL USD", commod: "PUT Gold", stocks: "PUT Growth", col: C.red },
                  { regime: "Dovish Fed / Weak DXY",    crypto: "CALL", forex: "PUT USD", commod: "CALL Gold", stocks: "CALL Growth", col: C.green },
                  { regime: "Risk-On Market",            crypto: "CALL", forex: "PUT JPY/CHF", commod: "PUT Gold", stocks: "CALL Tech", col: C.green },
                  { regime: "Risk-Off / Fear Spike",     crypto: "PUT",  forex: "CALL JPY/CHF", commod: "CALL Gold", stocks: "PUT Tech", col: C.red },
                  { regime: "Geopolitical Shock",        crypto: "PUT→CALL", forex: "CALL CHF/JPY", commod: "CALL Gold+Oil", stocks: "PUT all", col: C.yellow },
                ].map((r, i, arr) => (
                  <div key={i} style={{ padding: "12px 16px", borderBottom: i < arr.length - 1 ? `1px solid ${C.border}` : "none", background: `${r.col}05` }}>
                    <div style={{ color: r.col, fontSize: 10, fontWeight: 800, marginBottom: 6 }}>{r.regime}</div>
                    <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                      {[["₿ Crypto", r.crypto], ["FX Forex", r.forex], ["◆ Metals", r.commod], ["◉ Stocks", r.stocks]].map(([cat, sig]) => (
                        <div key={cat as string} style={{ background: "#ffffff08", border: `1px solid ${C.border}`, borderRadius: 6, padding: "3px 8px", fontSize: 9 }}>
                          <span style={{ color: C.dim }}>{cat as string}: </span>
                          <b style={{ color: C.text }}>{sig as string}</b>
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>

              {/* ── Upcoming Catalysts ───────────────────────────────────── */}
              <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 14, overflow: "hidden" }}>
                <div style={{ background: "#0a0f1a", padding: "12px 16px", borderBottom: `1px solid ${C.border}` }}>
                  <div style={{ color: C.dim, fontSize: 10, letterSpacing: "0.14em", fontWeight: 700 }}>UPCOMING CATALYSTS — MAY 2026</div>
                </div>
                {[
                  { date: "May 8",  event: "NFP Jobs Report 8:30AM",          impact: "MAXIMUM",   color: C.red },
                  { date: "May 11", event: "Senate returns — Warsh vote",      impact: "HIGH",      color: C.yellow },
                  { date: "May 12", event: "CPI April 8:30AM",                 impact: "HIGH",      color: C.yellow },
                  { date: "May 21", event: "CLARITY Act deadline",             impact: "CRYPTO HIGH", color: C.blue },
                  { date: "May 28", event: "PCE Inflation Data",               impact: "HIGH",      color: C.yellow },
                ].map((c, i, arr) => (
                  <div key={i} style={{ display: "flex", alignItems: "center", gap: 12, padding: "12px 16px", borderBottom: i < arr.length - 1 ? `1px solid ${C.border}` : "none", background: `${c.color}05` }}>
                    <div style={{ color: c.color, fontSize: 10, fontFamily: "monospace", fontWeight: 700, flexShrink: 0, minWidth: 44 }}>{c.date}</div>
                    <div style={{ flex: 1, color: C.text, fontSize: 12 }}>{c.event}</div>
                    <div style={{ background: `${c.color}20`, border: `1px solid ${c.color}50`, borderRadius: 6, padding: "3px 8px", flexShrink: 0 }}>
                      <span style={{ color: c.color, fontSize: 9, fontWeight: 700 }}>{c.impact}</span>
                    </div>
                  </div>
                ))}
              </div>

              {/* ── Execution Rules ──────────────────────────────────────── */}
              <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 14, overflow: "hidden" }}>
                <div style={{ background: "#0a0f1a", padding: "12px 16px", borderBottom: `1px solid ${C.border}` }}>
                  <div style={{ color: C.dim, fontSize: 10, letterSpacing: "0.14em", fontWeight: 700 }}>EXECUTION RULES</div>
                </div>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr" }}>
                  {[
                    { label: "Wait After News", value: "45–60s",  color: C.yellow },
                    { label: "Max Position",     value: "5% bal",  color: C.green },
                    { label: "Crypto Expiry",    value: "15 min",  color: C.blue },
                    { label: "Min Score",        value: "4 / 7",   color: C.red },
                  ].map((r, i) => (
                    <div key={i} style={{ padding: "16px 14px", textAlign: "center", borderRight: i % 2 === 0 ? `1px solid ${C.border}` : "none", borderBottom: i < 2 ? `1px solid ${C.border}` : "none", background: `${r.color}08` }}>
                      <div style={{ color: r.color, fontSize: 22, fontWeight: 900, marginBottom: 4 }}>{r.value}</div>
                      <div style={{ color: "#475569", fontSize: 10 }}>{r.label}</div>
                    </div>
                  ))}
                </div>
              </div>

              <div style={{ textAlign: "center", fontSize: 9, color: "#374151", lineHeight: 2 }}>
                LIVE DATA: COINGECKO + ALTERNATIVE.ME · 13 TOKENS · AUTO-REFRESH 90S<br />
                EDUCATIONAL ONLY · NOT FINANCIAL ADVICE
              </div>
            </>
          )}
        </div>
      </div>
    </>
  );
}
