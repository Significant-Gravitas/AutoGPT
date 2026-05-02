import { useState, useEffect, useCallback } from "react";
import Head from "next/head";

/* ── Color palette ───────────────────────────── */
const C = {
  bg: "#060a0e",
  card: "#0d1117",
  border: "#1e2d3d",
  green: "#00ff88",
  red: "#ff4560",
  yellow: "#f0c040",
  blue: "#38bdf8",
  text: "#e2e8f0",
  dim: "#64748b",
};

/* ── Token list ──────────────────────────────── */
const TOKENS = [
  { key: "btc", id: "bitcoin", symbol: "BTC/USD", role: "Primary macro asset" },
  { key: "eth", id: "ethereum", symbol: "ETH/USD", role: "Alt market proxy" },
  { key: "sol", id: "solana", symbol: "SOL/USD", role: "High-beta risk signal" },
  { key: "xrp", id: "ripple", symbol: "XRP/USD", role: "Liquidity indicator" },
  { key: "bnb", id: "binancecoin", symbol: "BNB/USD", role: "Exchange flow signal" },
];

/* ── Public API endpoints (no API key) ────────── */
const COINGECKO_BASE = "https://api.coingecko.com/api/v3";
const COINGECKO_SIMPLE = `${COINGECKO_BASE}/simple/price?ids=bitcoin,ethereum,solana,ripple,binancecoin&vs_currencies=usd&include_24hr_change=true`;
const COINGECKO_GLOBAL = `${COINGECKO_BASE}/global`;
const FEAR_GREED_URL = "https://api.alternative.me/fng/?limit=1";

/* ── Types ──────────────────────────────────── */
interface MarketData {
  btc_price: number | null;
  btc_chg: number | null;
  eth_price: number | null;
  eth_chg: number | null;
  sol_price: number | null;
  sol_chg: number | null;
  xrp_price: number | null;
  xrp_chg: number | null;
  bnb_price: number | null;
  bnb_chg: number | null;
  btc_dominance: number | null;
  total_mcap: string;
  fg_value: number;
  fg_label: string;
  oil_brent: string;
  fed_rate: string;
}

interface Factor {
  label: string;
  value: string;
  detail: string;
  status: "B" | "R" | "N";
}

interface Signal {
  label: string;
  color: string;
}

/* ── Format helpers ──────────────────────────── */
function fmtPrice(price: number | null, key: string): string {
  if (price == null) return "—";
  const n = Number(price);
  if (key === "xrp") return `$${n.toFixed(4)}`;
  if (n < 200) return `$${n.toFixed(2)}`;
  return `$${n.toLocaleString("en-US", { maximumFractionDigits: 2 })}`;
}

/* ── Signal logic ────────────────────────────── */
function tokenSignal(change24h: number | null): Signal {
  if (change24h == null) return { label: "WAIT", color: C.yellow };
  if (change24h > 0.5) return { label: "CALL", color: C.green };
  if (change24h < -0.5) return { label: "PUT", color: C.red };
  return { label: "WAIT", color: C.yellow };
}

function compositeBias(score: number): Signal {
  if (score >= 5) return { label: "CALL", color: C.green };
  if (score <= 2) return { label: "PUT", color: C.red };
  return { label: "WAIT", color: C.yellow };
}

/* ── Spinner component ──────────────────────── */
function Spin({ size = 20, color = C.green }: { size?: number; color?: string }) {
  return (
    <div
      style={{
        width: size,
        height: size,
        border: `2px solid #1e2d3d`,
        borderTop: `2px solid ${color}`,
        borderRadius: "50%",
        animation: "spin 0.7s linear infinite",
        flexShrink: 0,
      }}
    />
  );
}

/* ── Main component ──────────────────────────── */
export default function SignalEngine() {
  const [market, setMarket] = useState<MarketData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState<string | null>(null);

  const fetchMarketData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [priceRes, globalRes, fgRes] = await Promise.all([
        fetch(COINGECKO_SIMPLE),
        fetch(COINGECKO_GLOBAL),
        fetch(FEAR_GREED_URL),
      ]);

      if (!priceRes.ok || !globalRes.ok || !fgRes.ok) {
        throw new Error("One or more API endpoints returned an error");
      }

      const priceData = await priceRes.json() as Record<string, Record<string, number>>;
      const globalData = await globalRes.json() as { data: { market_cap_percentage: Record<string, number>; total_market_cap: Record<string, number> } };
      const fgData = await fgRes.json() as { data: Array<{ value: string; value_classification: string }> };

      const parsed: MarketData = {
        btc_price: priceData.bitcoin?.usd ?? null,
        btc_chg: priceData.bitcoin?.usd_24h_change ?? null,
        eth_price: priceData.ethereum?.usd ?? null,
        eth_chg: priceData.ethereum?.usd_24h_change ?? null,
        sol_price: priceData.solana?.usd ?? null,
        sol_chg: priceData.solana?.usd_24h_change ?? null,
        xrp_price: priceData.ripple?.usd ?? null,
        xrp_chg: priceData.ripple?.usd_24h_change ?? null,
        bnb_price: priceData.binancecoin?.usd ?? null,
        bnb_chg: priceData.binancecoin?.usd_24h_change ?? null,
        btc_dominance: globalData.data?.market_cap_percentage?.btc ?? null,
        total_mcap: ((globalData.data?.total_market_cap?.usd ?? 0) / 1e12).toFixed(2),
        fg_value: parseInt(fgData.data?.[0]?.value ?? "50", 10),
        fg_label: fgData.data?.[0]?.value_classification ?? "—",
        oil_brent: "N/A",
        fed_rate: "3.50–3.75%",
      };

      setMarket(parsed);
      setLastUpdate(new Date().toLocaleTimeString());
    } catch (err) {
      console.error(err);
      setError("Failed to fetch live data. Check your connection.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void fetchMarketData();
    const interval = setInterval(() => void fetchMarketData(), 90 * 1000);
    return () => clearInterval(interval);
  }, [fetchMarketData]);

  /* ── Compute signals ────────────────────────── */
  let score = 0;
  const factors: Factor[] = [];

  if (market) {
    const { fg_value, btc_chg, eth_chg, btc_dominance } = market;

    if (fg_value <= 25) score += 2;
    else if (fg_value <= 45) score += 1;
    else if (fg_value >= 75) score -= 1;

    if (btc_chg != null) {
      if (btc_chg > 4) score += 2;
      else if (btc_chg > 1) score += 1;
      else if (btc_chg < -4) score -= 2;
      else if (btc_chg < -1) score -= 1;
    }

    if (eth_chg != null) {
      if (eth_chg > 3) score += 1;
      else if (eth_chg < -3) score -= 1;
    }

    if (btc_dominance != null) {
      if (btc_dominance > 58) score -= 1;
      else if (btc_dominance < 45) score += 1;
    }

    const btcUp = (btc_chg ?? 0) > 0;
    const ethUp = (eth_chg ?? 0) > 0;
    const aligned = (btcUp && ethUp) || (!btcUp && !ethUp);
    if (aligned && Math.abs(btc_chg ?? 0) > 1) score += 1;

    score = Math.max(0, Math.min(7, score));

    factors.push({
      label: "Fear & Greed",
      value: `${fg_value} — ${market.fg_label}`,
      detail:
        fg_value <= 25
          ? "Extreme fear — historically a contrarian buy zone"
          : fg_value <= 45
          ? "Fear zone — possible accumulation"
          : fg_value >= 75
          ? "Greed — reversal risk elevated"
          : "Neutral — awaiting catalyst",
      status: fg_value <= 45 ? "B" : fg_value >= 75 ? "R" : "N",
    });

    factors.push({
      label: "BTC 24h Momentum",
      value: btc_chg != null ? `${btc_chg > 0 ? "+" : ""}${btc_chg.toFixed(2)}%` : "—",
      detail:
        (btc_chg ?? 0) > 1
          ? "Positive momentum — trend following"
          : (btc_chg ?? 0) < -1
          ? "Negative momentum — caution"
          : "Sideways — low conviction",
      status: (btc_chg ?? 0) > 1 ? "B" : (btc_chg ?? 0) < -1 ? "R" : "N",
    });

    factors.push({
      label: "ETH / Alt Proxy",
      value: eth_chg != null ? `${eth_chg > 0 ? "+" : ""}${eth_chg.toFixed(2)}%` : "—",
      detail:
        (eth_chg ?? 0) > 3
          ? "Alts strong — broad participation"
          : (eth_chg ?? 0) < -3
          ? "Alts weak — risk-off"
          : "Mild alt action",
      status: (eth_chg ?? 0) > 3 ? "B" : (eth_chg ?? 0) < -3 ? "R" : "N",
    });

    factors.push({
      label: "BTC Dominance",
      value: btc_dominance != null ? `${btc_dominance.toFixed(1)}%` : "—",
      detail:
        (btc_dominance ?? 0) > 58
          ? "Bitcoin Season — capital concentrated in BTC"
          : (btc_dominance ?? 0) < 45
          ? "Alt Season — capital rotating to alts"
          : "Balanced — watch for rotation",
      status: (btc_dominance ?? 0) > 58 ? "R" : (btc_dominance ?? 0) < 45 ? "B" : "N",
    });

    const crossAligned =
      ((btc_chg ?? 0) > 0 && (eth_chg ?? 0) > 0) ||
      ((btc_chg ?? 0) < 0 && (eth_chg ?? 0) < 0);
    factors.push({
      label: "Cross-Asset Confirm",
      value: crossAligned ? "ALIGNED" : "DIVERGENT",
      detail: "BTC & ETH moving together → conviction",
      status: crossAligned ? "B" : "N",
    });
  }

  const bias = market ? compositeBias(score) : { label: "—", color: C.text };
  const scoreColor = score >= 5 ? C.green : score <= 2 ? C.red : C.yellow;

  return (
    <>
      <Head>
        <title>Signal Engine · Crypto Dashboard</title>
        <meta name="description" content="Live crypto signal engine with real-time prices and sentiment" />
      </Head>
      <div
        style={{
          background: C.bg,
          minHeight: "100vh",
          color: C.text,
          fontFamily: "'Courier New', monospace",
          paddingBottom: 52,
        }}
      >
        <style>{`
          @keyframes spin { to { transform: rotate(360deg); } }
          * { box-sizing: border-box; margin: 0; padding: 0; }
        `}</style>

        {/* Header */}
        <div
          style={{
            background: "#0a0f1a",
            borderBottom: `1px solid ${C.border}`,
            padding: "14px 16px",
            position: "sticky",
            top: 0,
            zIndex: 10,
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <div>
              <div style={{ color: C.green, fontSize: 15, fontWeight: 900, letterSpacing: "0.05em" }}>
                ⬡ SIGNAL ENGINE
              </div>
              <div style={{ color: C.dim, fontSize: 9, letterSpacing: "0.14em", marginTop: 2 }}>
                LIVE PUBLIC API · POCKET OPTION
              </div>
            </div>
            <button
              onClick={() => void fetchMarketData()}
              disabled={loading}
              style={{
                background: `${C.green}15`,
                border: `1px solid ${loading ? C.dim : C.green}`,
                color: loading ? C.dim : C.green,
                padding: "10px 18px",
                borderRadius: 8,
                fontSize: 12,
                fontWeight: 700,
                cursor: loading ? "not-allowed" : "pointer",
                fontFamily: "monospace",
                display: "flex",
                alignItems: "center",
                gap: 8,
              }}
            >
              {loading ? (
                <>
                  <Spin size={14} />
                  &nbsp;Refreshing
                </>
              ) : (
                "↺ Refresh Now"
              )}
            </button>
          </div>
          <div style={{ color: C.dim, fontSize: 9, marginTop: 5 }}>
            {lastUpdate
              ? `Last update: ${lastUpdate} · Auto-refresh every 90s`
              : "Initializing live connection..."}
          </div>
        </div>

        <div style={{ padding: "14px", display: "flex", flexDirection: "column", gap: 14 }}>
          {/* Error banner */}
          {error && (
            <div
              style={{
                background: "#ff456015",
                border: `1px solid ${C.red}50`,
                borderRadius: 10,
                padding: 16,
                color: C.red,
                fontSize: 13,
                textAlign: "center",
              }}
            >
              {error}
              <button
                onClick={() => void fetchMarketData()}
                style={{
                  marginLeft: 10,
                  background: "transparent",
                  border: `1px solid ${C.red}`,
                  color: C.red,
                  borderRadius: 6,
                  padding: "4px 12px",
                  cursor: "pointer",
                  fontSize: 12,
                }}
              >
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

          {market && (
            <>
              {/* Overall Signal */}
              <div
                style={{
                  background: `${bias.color}12`,
                  border: `2px solid ${bias.color}50`,
                  borderRadius: 14,
                  padding: "22px 16px",
                  textAlign: "center",
                  position: "relative",
                  overflow: "hidden",
                }}
              >
                <div
                  style={{
                    position: "absolute",
                    inset: 0,
                    background: `radial-gradient(ellipse at center,${bias.color}10,transparent 70%)`,
                    pointerEvents: "none",
                  }}
                />
                <div style={{ color: C.dim, fontSize: 9, letterSpacing: "0.2em", marginBottom: 10 }}>
                  POCKET OPTION · OVERALL SIGNAL
                </div>
                <div
                  style={{
                    fontSize: 66,
                    fontWeight: 900,
                    color: bias.color,
                    lineHeight: 1,
                    textShadow: `0 0 30px ${bias.color}60`,
                  }}
                >
                  {bias.label}
                </div>
                <div
                  style={{ color: bias.color, fontSize: 11, marginTop: 8, letterSpacing: "0.1em", opacity: 0.85 }}
                >
                  {bias.label === "CALL"
                    ? "Conditions favor LONG entries"
                    : bias.label === "PUT"
                    ? "Conditions favor SHORT entries"
                    : "Mixed signals — stand down"}
                </div>
                <div style={{ margin: "14px auto 0", maxWidth: 280 }}>
                  <div style={{ background: "#0a1018", borderRadius: 6, height: 8, overflow: "hidden" }}>
                    <div
                      style={{
                        width: `${(score / 7) * 100}%`,
                        height: "100%",
                        background: scoreColor,
                        boxShadow: `0 0 8px ${scoreColor}60`,
                        borderRadius: 6,
                        transition: "width 0.8s ease",
                      }}
                    />
                  </div>
                  <div style={{ display: "flex", justifyContent: "space-between", marginTop: 4 }}>
                    <span style={{ color: C.dim, fontSize: 9 }}>7-Factor Score</span>
                    <span style={{ color: scoreColor, fontSize: 10, fontWeight: 800 }}>{score} / 7</span>
                  </div>
                </div>
                {/* Macro context pills */}
                <div
                  style={{ display: "flex", gap: 8, justifyContent: "center", marginTop: 14, flexWrap: "wrap" }}
                >
                  {[
                    { label: "F&G", val: `${market.fg_value} · ${market.fg_label}`, col: C.yellow },
                    {
                      label: "Dom",
                      val: market.btc_dominance != null ? `${market.btc_dominance.toFixed(1)}%` : "—",
                      col: C.yellow,
                    },
                    { label: "Mcap", val: `$${market.total_mcap}T`, col: C.blue },
                    { label: "Oil", val: market.oil_brent, col: C.red },
                  ].map((p, i) => (
                    <div
                      key={i}
                      style={{
                        background: "#ffffff0a",
                        border: `1px solid ${C.border}`,
                        borderRadius: 20,
                        padding: "5px 12px",
                        fontSize: 10,
                      }}
                    >
                      <span style={{ color: C.dim }}>{p.label}: </span>
                      <b style={{ color: p.col }}>{p.val}</b>
                    </div>
                  ))}
                </div>
              </div>

              {/* Token Watchlist */}
              <div
                style={{
                  background: C.card,
                  border: `1px solid ${C.border}`,
                  borderRadius: 14,
                  overflow: "hidden",
                }}
              >
                <div
                  style={{
                    background: "#0a0f1a",
                    padding: "12px 16px",
                    borderBottom: `1px solid ${C.border}`,
                  }}
                >
                  <div style={{ color: C.dim, fontSize: 10, letterSpacing: "0.14em", fontWeight: 700 }}>
                    TOKEN WATCHLIST — LIVE PRICES
                  </div>
                </div>
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "1.8fr 1.4fr 72px 68px",
                    padding: "8px 16px",
                    borderBottom: `1px solid ${C.border}`,
                    background: "#0a0f1a80",
                  }}
                >
                  {["TOKEN", "PRICE", "24H", "SIGNAL"].map((h) => (
                    <div
                      key={h}
                      style={{ color: "#475569", fontSize: 9, letterSpacing: "0.12em", fontWeight: 700 }}
                    >
                      {h}
                    </div>
                  ))}
                </div>
                {TOKENS.map((t, i) => {
                  const price = market[`${t.key}_price` as keyof MarketData] as number | null;
                  const chg = market[`${t.key}_chg` as keyof MarketData] as number | null;
                  const sig = tokenSignal(chg);
                  const changeColor = (chg ?? 0) >= 0 ? C.green : C.red;
                  return (
                    <div
                      key={t.key}
                      style={{
                        display: "grid",
                        gridTemplateColumns: "1.8fr 1.4fr 72px 68px",
                        padding: "15px 16px",
                        alignItems: "center",
                        borderBottom: i < TOKENS.length - 1 ? `1px solid ${C.border}` : "none",
                        background: `${sig.color}06`,
                      }}
                    >
                      <div>
                        <div style={{ color: C.text, fontSize: 14, fontWeight: 800 }}>{t.symbol}</div>
                        <div style={{ color: "#475569", fontSize: 9, marginTop: 2 }}>{t.role}</div>
                      </div>
                      <div
                        style={{
                          color: C.text,
                          fontSize: 15,
                          fontWeight: 800,
                          fontFamily: "monospace",
                        }}
                      >
                        {fmtPrice(price, t.key)}
                      </div>
                      <div>
                        <div
                          style={{
                            color: changeColor,
                            fontSize: 12,
                            fontWeight: 700,
                            fontFamily: "monospace",
                          }}
                        >
                          {chg != null ? `${chg > 0 ? "+" : ""}${chg.toFixed(2)}%` : "—"}
                        </div>
                        <div style={{ color: changeColor, fontSize: 10, marginTop: 2 }}>
                          {(chg ?? 0) >= 0 ? "▲" : "▼"}
                        </div>
                      </div>
                      <div
                        style={{
                          background: `${sig.color}20`,
                          border: `1px solid ${sig.color}55`,
                          borderRadius: 8,
                          padding: "7px 0",
                          textAlign: "center",
                        }}
                      >
                        <div style={{ color: sig.color, fontSize: 11, fontWeight: 900 }}>{sig.label}</div>
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* 7-Factor Breakdown */}
              <div
                style={{
                  background: C.card,
                  border: `1px solid ${C.border}`,
                  borderRadius: 14,
                  overflow: "hidden",
                }}
              >
                <div
                  style={{
                    background: "#0a0f1a",
                    padding: "12px 16px",
                    borderBottom: `1px solid ${C.border}`,
                  }}
                >
                  <div style={{ color: C.dim, fontSize: 10, letterSpacing: "0.14em", fontWeight: 700 }}>
                    7-FACTOR SIGNAL BREAKDOWN
                  </div>
                </div>
                {factors.map((f, i) => (
                  <div
                    key={i}
                    style={{
                      display: "flex",
                      alignItems: "flex-start",
                      gap: 12,
                      padding: "13px 16px",
                      borderBottom: i < factors.length - 1 ? `1px solid ${C.border}` : "none",
                      background: `${f.status === "B" ? C.green : f.status === "R" ? C.red : C.yellow}06`,
                    }}
                  >
                    <span
                      style={{
                        color: f.status === "B" ? C.green : f.status === "R" ? C.red : C.yellow,
                        fontSize: 13,
                        width: 14,
                        flexShrink: 0,
                        marginTop: 1,
                        textAlign: "center",
                      }}
                    >
                      {f.status === "B" ? "▲" : f.status === "R" ? "▼" : "━"}
                    </span>
                    <div style={{ flex: 1 }}>
                      <div style={{ color: C.text, fontSize: 12, fontWeight: 700 }}>{f.label}</div>
                      <div style={{ color: C.dim, fontSize: 10, marginTop: 2 }}>{f.detail}</div>
                    </div>
                    <span
                      style={{
                        color: f.status === "B" ? C.green : f.status === "R" ? C.red : C.yellow,
                        fontSize: 11,
                        fontFamily: "monospace",
                        fontWeight: 700,
                        flexShrink: 0,
                        marginTop: 1,
                      }}
                    >
                      {f.value}
                    </span>
                  </div>
                ))}
              </div>

              {/* Upcoming Catalysts */}
              <div
                style={{
                  background: C.card,
                  border: `1px solid ${C.border}`,
                  borderRadius: 14,
                  overflow: "hidden",
                }}
              >
                <div
                  style={{
                    background: "#0a0f1a",
                    padding: "12px 16px",
                    borderBottom: `1px solid ${C.border}`,
                  }}
                >
                  <div style={{ color: C.dim, fontSize: 10, letterSpacing: "0.14em", fontWeight: 700 }}>
                    UPCOMING CATALYSTS — MAY 2026
                  </div>
                </div>
                {[
                  { date: "May 8", event: "NFP Jobs Report 8:30AM", impact: "MAXIMUM", color: C.red },
                  { date: "May 11", event: "Senate returns — Warsh vote", impact: "HIGH", color: C.yellow },
                  { date: "May 12", event: "CPI April 8:30AM", impact: "HIGH", color: C.yellow },
                  { date: "May 21", event: "CLARITY Act deadline", impact: "HIGH", color: C.blue },
                  { date: "May 28", event: "PCE Inflation Data", impact: "HIGH", color: C.yellow },
                ].map((c, i, arr) => (
                  <div
                    key={i}
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: 12,
                      padding: "12px 16px",
                      borderBottom: i < arr.length - 1 ? `1px solid ${C.border}` : "none",
                      background: `${c.color}05`,
                    }}
                  >
                    <div
                      style={{
                        color: c.color,
                        fontSize: 10,
                        fontFamily: "monospace",
                        fontWeight: 700,
                        flexShrink: 0,
                        minWidth: 44,
                      }}
                    >
                      {c.date}
                    </div>
                    <div style={{ flex: 1, color: C.text, fontSize: 12 }}>{c.event}</div>
                    <div
                      style={{
                        background: `${c.color}20`,
                        border: `1px solid ${c.color}50`,
                        borderRadius: 6,
                        padding: "3px 8px",
                        flexShrink: 0,
                      }}
                    >
                      <span style={{ color: c.color, fontSize: 9, fontWeight: 700 }}>{c.impact}</span>
                    </div>
                  </div>
                ))}
              </div>

              {/* Execution Rules */}
              <div
                style={{
                  background: C.card,
                  border: `1px solid ${C.border}`,
                  borderRadius: 14,
                  overflow: "hidden",
                }}
              >
                <div
                  style={{
                    background: "#0a0f1a",
                    padding: "12px 16px",
                    borderBottom: `1px solid ${C.border}`,
                  }}
                >
                  <div style={{ color: C.dim, fontSize: 10, letterSpacing: "0.14em", fontWeight: 700 }}>
                    EXECUTION RULES
                  </div>
                </div>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr" }}>
                  {[
                    { label: "Wait After News", value: "45–60s", color: C.yellow },
                    { label: "Max Position", value: "5% bal", color: C.green },
                    { label: "Crypto Expiry", value: "15 min", color: C.blue },
                    { label: "Min Score", value: "4 / 7", color: C.red },
                  ].map((r, i) => (
                    <div
                      key={i}
                      style={{
                        padding: "16px 14px",
                        textAlign: "center",
                        borderRight: i % 2 === 0 ? `1px solid ${C.border}` : "none",
                        borderBottom: i < 2 ? `1px solid ${C.border}` : "none",
                        background: `${r.color}08`,
                      }}
                    >
                      <div style={{ color: r.color, fontSize: 22, fontWeight: 900, marginBottom: 4 }}>
                        {r.value}
                      </div>
                      <div style={{ color: "#475569", fontSize: 10 }}>{r.label}</div>
                    </div>
                  ))}
                </div>
              </div>

              <div style={{ textAlign: "center", fontSize: 9, color: "#374151", lineHeight: 2 }}>
                LIVE DATA: COINGECKO + ALTERNATIVE.ME · AUTO-REFRESH 90S
                <br />
                EDUCATIONAL ONLY · NOT FINANCIAL ADVICE
              </div>
            </>
          )}
        </div>
      </div>
    </>
  );
}
