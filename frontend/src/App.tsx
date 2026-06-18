import { DeckGL } from "@deck.gl/react";
import { LineLayer } from "@deck.gl/layers";
import { SimpleMeshLayer } from "deck.gl";
import { OrbitView } from "@deck.gl/core";
import type { OrbitViewState } from "@deck.gl/core";
import { SphereGeometry } from "@luma.gl/engine";
import { useState, useMemo, useEffect, useCallback } from "react";

// --- Shared sphere geometry for 3D graph nodes ---
const SPHERE_GEOM = new SphereGeometry({ radius: 1, nlat: 32, nlong: 32 });

// ======================================================================
// Types
// ======================================================================

interface StatsResponse {
  utterances: number; conversations: number;
  agents: { agent: string; count: number }[];
  months: { month: string; count: number }[];
}

interface AnalysisResponse {
  stats: {
    utterance_count: number; conversation_count: number;
    avg_words: number; median_words: number; max_words: number; avg_chars: number;
    agent_distribution: Record<string, number>;
    month_distribution: Record<string, number>;
    properties: Record<string, number>;
    style_warnings: { source: string; message: string; excerpt: string }[];
  };
}

interface UtteranceItem {
  id: string; source_agent: string; conversation_id: string;
  turn_index: number; timestamp: string | null; text: string; source_path: string;
}

interface GraphNode {
  id: string; text: string; source_agent: string;
  x: number; y: number; z: number; cluster_id: number;
}

interface GraphEdge { source: string; target: string; similarity: number; }

interface GraphResponse { nodes: GraphNode[]; edges: GraphEdge[]; }

interface UtterancesResponse { utterances: UtteranceItem[] }

interface CoachingItem {
  utterance_id: string; text: string; source_agent: string;
  clarity_score: number; interaction_style: string;
  feedback: {
    intent?: string; what_worked?: string; could_improve?: string;
    better_prompt?: string; hidden_tip?: string; knowledge_gap?: string;
    clarity_score?: number; interaction_style?: string;
  };
}

interface CoachingResponse { coaching: CoachingItem[] }

interface CoachingSummary {
  total_coached: number; avg_clarity: number;
  dominant_style: string; common_issues: string[]; top_tips: string[];
}

type Page = "graph" | "dashboard" | "coaching";

// ======================================================================
// Linear/Bento Design Tokens
// ======================================================================

const AGENT_COLORS: Record<string, [number, number, number]> = {
  reasonix: [96, 165, 250],
  opencode: [74, 222, 128],
  codex: [250, 204, 21],
  "claude-code": [251, 146, 60],
  codewhale: [167, 139, 250],
  "kilo-code": [216, 180, 254],
  openclaw: [244, 114, 182],
};
const AGENT_FALLBACK: [number, number, number] = [148, 163, 184];

function agentColor(agent: string): [number, number, number] {
  const key = agent.toLowerCase();
  for (const [k, v] of Object.entries(AGENT_COLORS)) {
    if (key.includes(k)) return v;
  }
  return AGENT_FALLBACK;
}

function agentHex(agent: string): string {
  const c = agentColor(agent);
  return `#${c.map(v => v.toString(16).padStart(2, "0")).join("")}`;
}

function similarityHeat(sim: number): [number, number, number] {
  // Neon: dim cyan → bright cyan → white
  const t = Math.max(0, Math.min(1, (sim - 0.3) / 0.7));
  const r = Math.round(0 + t * 200);
  const g = Math.round(180 + t * 75);
  const b = Math.round(200 + t * 55);
  return [r, g, b];
}

// ======================================================================
// Inline Markdown Renderer
// ======================================================================

function renderInlineMarkdown(text: string): string {
  return text
    // Escape HTML first
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    // Inline code: `code`
    .replace(/`([^`]+)`/g, '<code class="px-1 py-0.5 rounded bg-[#1e1e2c] text-[#5eead4] text-[11px] font-mono">$1</code>')
    // Bold: **text**
    .replace(/\*\*([^*]+)\*\*/g, '<b class="text-[#e4e4e7]">$1</b>')
    // Italic: *text*
    .replace(/\*([^*]+)\*/g, '<i>$1</i>');
}

function SafeMarkdown({ text }: { text: string }) {
  const html = useMemo(() => renderInlineMarkdown(text), [text]);
  return <span dangerouslySetInnerHTML={{ __html: html }} />;
}

function Sidebar({ page, setPage, stats, coachingSummary }: {
  page: Page; setPage: (p: Page) => void;
  stats: StatsResponse | null; coachingSummary: CoachingSummary | null;
}) {
  const links: { page: Page; label: string; icon: string }[] = [
    { page: "graph", label: "Knowledge Graph", icon: "◉" },
    { page: "dashboard", label: "Analytics", icon: "▤" },
    { page: "coaching", label: "AI Coaching", icon: "✦" },
  ];
  return (
    <aside className="w-[220px] shrink-0 h-screen bg-[#0b0b12] border-r border-[#1e1e2c] flex flex-col select-none">
      <div className="px-5 py-5 border-b border-[#1e1e2c]">
        <h1 className="text-base font-semibold text-[#e4e4e7] tracking-tight">Agentrace</h1>
        <p className="text-[11px] text-[#71717a] mt-0.5">AI Conversation Analytics</p>
      </div>
      <nav className="flex-1 px-3 py-3 space-y-0.5">
        {links.map(l => (
          <button key={l.page}
            onClick={() => setPage(l.page)}
            className={`w-full text-left px-3 py-2 rounded-lg text-sm transition-all flex items-center gap-2.5 ${
              page === l.page
                ? "bg-[#1e1e2c] text-[#e4e4e7] font-medium"
                : "text-[#a1a1aa] hover:text-[#e4e4e7] hover:bg-[#14141c]"
            }`}>
            <span className="text-xs w-4 text-center">{l.icon}</span>
            {l.label}
          </button>
        ))}
      </nav>
      <div className="px-4 py-4 border-t border-[#1e1e2c] space-y-1.5 text-[11px]">
        <div className="flex justify-between text-[#71717a]">
          <span>Utterances</span>
          <span className="text-[#a1a1aa] font-mono">{stats?.utterances ?? "—"}</span>
        </div>
        <div className="flex justify-between text-[#71717a]">
          <span>Conversations</span>
          <span className="text-[#a1a1aa] font-mono">{stats?.conversations ?? "—"}</span>
        </div>
        {coachingSummary && (
          <div className="flex justify-between text-[#71717a]">
            <span>Avg Clarity</span>
            <span className="text-[#a1a1aa] font-mono">{coachingSummary.avg_clarity.toFixed(1)}/5</span>
          </div>
        )}
      </div>
    </aside>
  );
}

// ======================================================================
// Bento Card Shell
// ======================================================================

function Card({ title, children, className }: { title?: string; children: React.ReactNode; className?: string }) {
  return (
    <div className={`bg-[#0f0f18] rounded-xl border border-[#1e1e2c] p-5 ${className ?? ""}`}>
      {title && <h3 className="text-xs font-semibold text-[#71717a] uppercase tracking-widest mb-3">{title}</h3>}
      {children}
    </div>
  );
}

// ======================================================================
// Bar Chart (mini inline)
// ======================================================================

function MiniBar({ label, value, max, color }: { label: string; value: number; max: number; color: string }) {
  const pct = max > 0 ? (value / max) * 100 : 0;
  return (
    <div className="flex items-center gap-2 text-xs">
      <span className="w-20 text-[#a1a1aa] truncate">{label}</span>
      <div className="flex-1 h-2 bg-[#1e1e2c] rounded-full overflow-hidden">
        <div className="h-full rounded-full transition-all duration-500" style={{ width: `${pct}%`, background: color }} />
      </div>
      <span className="w-8 text-right text-[#e4e4e7] font-mono tabular-nums">{value}</span>
    </div>
  );
}

// ======================================================================
// Coaching Detail Panel
// ======================================================================

function CoachingDetail({ item, onClose }: { item: CoachingItem; onClose: () => void }) {
  const f = item.feedback;
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm" onClick={onClose}>
      <div className="bg-[#0f0f18] rounded-xl border border-[#1e1e2c] p-6 max-w-lg w-full mx-4 max-h-[80vh] overflow-y-auto shadow-2xl" onClick={e => e.stopPropagation()}>
        <div className="flex justify-between items-start mb-4">
          <div>
            <span className="px-2 py-0.5 rounded text-[11px] font-medium" style={{ background: agentHex(item.source_agent) + "22", color: agentHex(item.source_agent) }}>{item.source_agent}</span>
            <span className="ml-2 text-xs text-[#71717a]">Clarity {item.clarity_score}/5</span>
          </div>
          <button onClick={onClose} className="text-[#71717a] hover:text-[#e4e4e7] text-lg leading-none">&times;</button>
        </div>
        <p className="text-sm text-[#e4e4e7] leading-relaxed mb-4 border-l-2 border-[#5eead4] pl-3">{item.text}</p>
        {f.what_worked && <Section label="What Worked" text={f.what_worked} color="#4ade80" />}
        {f.could_improve && <Section label="Could Improve" text={f.could_improve} color="#fbbf24" />}
        {f.better_prompt && <Section label="Better Prompt" text={f.better_prompt} color="#60a5fa" />}
        {f.hidden_tip && <Section label="Hidden Tip" text={f.hidden_tip} color="#c084fc" />}
        {f.knowledge_gap && <Section label="Knowledge Gap" text={f.knowledge_gap} color="#fb923c" />}
      </div>
    </div>
  );
}

function Section({ label, text, color }: { label: string; text: string; color: string }) {
  return (
    <div className="mb-3">
      <span className="text-[11px] font-semibold uppercase tracking-wider" style={{ color }}>{label}</span>
      <p className="text-xs text-[#a1a1aa] mt-0.5 leading-relaxed"><SafeMarkdown text={text} /></p>
    </div>
  );
}

// ======================================================================
// Main App
// ======================================================================

export default function App() {
  const [page, setPage] = useState<Page>("graph");
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [analysis, setAnalysis] = useState<AnalysisResponse | null>(null);
  const [utterances, setUtterances] = useState<UtteranceItem[]>([]);
  const [graphData, setGraphData] = useState<GraphResponse | null>(null);
  const [coaching, setCoaching] = useState<CoachingItem[]>([]);
  const [coachingSummary, setCoachingSummary] = useState<CoachingSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [tooltip, setTooltip] = useState<string | null>(null);
  const [tooltipPos, setTooltipPos] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [selectedUtterance, setSelectedUtterance] = useState<UtteranceItem | null>(null);
  const [selectedCoaching, setSelectedCoaching] = useState<CoachingItem | null>(null);
  const [viewState, setViewState] = useState<OrbitViewState>({
    target: [0, 0, 0],
    rotationX: -35,
    rotationOrbit: 30,
    zoom: 0.8,
    minZoom: 0.1,
    maxZoom: 5,
  });

  const fetchData = useCallback(async () => {
    try {
      const [sRes, aRes, uRes, gRes, cRes, csRes] = await Promise.all([
        fetch("/api/v1/stats"), fetch("/api/v1/analysis"),
        fetch("/api/v1/utterances"), fetch("/api/v1/graph"),
        fetch("/api/v1/coaching"), fetch("/api/v1/coaching/summary"),
      ]);
      if (sRes.ok) setStats(await sRes.json());
      if (aRes.ok) setAnalysis(await aRes.json());
      if (uRes.ok) { const d: UtterancesResponse = await uRes.json(); setUtterances(d.utterances); }
      if (gRes.ok) { const d: GraphResponse = await gRes.json(); if (d.nodes?.length) setGraphData(d); }
      if (cRes.ok) { const d: CoachingResponse = await cRes.json(); setCoaching(d.coaching ?? []); }
      if (csRes.ok) setCoachingSummary(await csRes.json());
      setError(null);
    } catch { setError("Cannot reach API. Start with: agentrace serve"); }
    finally { setLoading(false); }
  }, []);

  useEffect(() => { fetchData(); }, [fetchData]);

  // Build node lookup for edges
  const nodeMap = useMemo(() => {
    const m = new Map<string, GraphNode>();
    if (graphData?.nodes) for (const n of graphData.nodes) m.set(n.id, n);
    return m;
  }, [graphData]);

  // Utterance lookup
  const uttMap = useMemo(() => {
    const m = new Map<string, UtteranceItem>();
    for (const u of utterances) m.set(u.id, u);
    return m;
  }, [utterances]);

  // ================================================================
  // Graph View
  // ================================================================

  const graphPoints = useMemo(() => {
    if (!graphData?.nodes) return [];
    const scale = 80;
    // Find z-range for gradient
    const zs = graphData.nodes.map(n => n.z);
    const zMin = Math.min(...zs);
    const zMax = Math.max(...zs);
    const zRange = zMax - zMin || 1;

    return graphData.nodes.map((gn) => {
      const u = uttMap.get(gn.id);
      const wordCount = gn.text.split(/\s+/).length;
      // Tiny neon dots — size barely varies with word count
      const radius = 0.15 + Math.log2(Math.max(1, wordCount)) * 0.15;
      // Z-gradient: cyan (#00e5ff) at top → indigo (#6366f1) at bottom
      const t = (gn.z - zMin) / zRange;
      const r = Math.round(0 + t * 99);
      const g = Math.round(229 + t * (102 - 229));
      const b = Math.round(255 + t * (241 - 255));
      return {
        position: [gn.x * scale, gn.y * scale, gn.z * scale] as [number, number, number],
        size: radius,
        color: [r, g, b, 255] as [number, number, number, number],
        id: gn.id,
        text: gn.text,
        source_agent: gn.source_agent,
        cluster_id: gn.cluster_id,
        utterance: u ?? null,
      };
    });
  }, [graphData, uttMap]);

  // Similarity edges
  const similarityEdgeData = useMemo(() => {
    if (!graphData?.edges) return [];
    return graphData.edges
      .filter(e => nodeMap.has(e.source) && nodeMap.has(e.target))
      .map(e => {
        const s = nodeMap.get(e.source)!;
        const t = nodeMap.get(e.target)!;
        const scale = 80;
        return {
          sourcePosition: [s.x * scale, s.y * scale, s.z * scale] as [number, number, number],
          targetPosition: [t.x * scale, t.y * scale, t.z * scale] as [number, number, number],
          similarity: e.similarity,
          color: similarityHeat(e.similarity),
          sourceText: s.text,
          targetText: t.text,
        };
      });
  }, [graphData, nodeMap]);

  const graphLayers = useMemo(() => {
    const ls: any[] = [];
    if (similarityEdgeData.length > 0) {
      ls.push(new LineLayer({
        id: "similarity-edges", data: similarityEdgeData,
        getSourcePosition: (d: any) => d.sourcePosition,
        getTargetPosition: (d: any) => d.targetPosition,
        getColor: (d: any) => d.color,
        getWidth: (d: any) => 0.2 + d.similarity * 2,
        opacity: 0.5,
        widthMinPixels: 0.3,
        widthMaxPixels: 6,
        onHover: (info: any) => {
          if (info.object && info.x !== undefined) {
            const obj = info.object;
            const srcShort = obj.sourceText.length > 80 ? `${obj.sourceText.slice(0, 80)}…` : obj.sourceText;
            const tgtShort = obj.targetText.length > 80 ? `${obj.targetText.slice(0, 80)}…` : obj.targetText;
            setTooltip(`Similarity: ${obj.similarity.toFixed(3)}\n"${srcShort}"\n↔ "${tgtShort}"`);
            setTooltipPos({ x: info.x + 14, y: info.y + 14 });
            document.body.style.cursor = "pointer";
          } else { setTooltip(null); document.body.style.cursor = "grab"; }
        },
      }));
    }
    // Glow halo: large transparent spheres behind nodes
    ls.push(new SimpleMeshLayer({
      id: "graph-nodes-glow",
      data: graphPoints,
      mesh: SPHERE_GEOM,
      getPosition: (d: any) => d.position,
      getColor: (d: any) => [d.color[0], d.color[1], d.color[2], 40] as [number, number, number, number],
      getOrientation: [0, 0, 0] as [number, number, number],
      getScale: (d: any) => [d.size * 4, d.size * 4, d.size * 4] as [number, number, number],
      sizeScale: 0.5,
      material: false,
      pickable: false,
    }));
    ls.push(new SimpleMeshLayer({
      id: "graph-nodes",
      data: graphPoints,
      mesh: SPHERE_GEOM,
      getPosition: (d: any) => d.position,
      getColor: (d: any) => d.color,
      getOrientation: [0, 0, 0] as [number, number, number],
      getScale: (d: any) => [d.size, d.size, d.size] as [number, number, number],
      sizeScale: 0.5,
      material: false,
      pickable: true,
      onClick: (info: any) => {
        if (info.object?.utterance) setSelectedUtterance(info.object.utterance);
      },
      onHover: (info: any) => {
        if (info.object && info.x !== undefined) {
          const obj = info.object;
          const preview = obj.text.length > 200
            ? `${obj.text.slice(0, 200)}…`
            : obj.text;
          setTooltip(`${obj.source_agent} · turn ${obj.utterance?.turn_index ?? "?"}\n${preview}`);
          setTooltipPos({ x: info.x + 14, y: info.y + 14 });
          document.body.style.cursor = "pointer";
        } else { setTooltip(null); document.body.style.cursor = "grab"; }
      },
    }));
    return ls;
  }, [graphPoints, similarityEdgeData]);

  // ================================================================
  // Render helpers
  // ================================================================

  const agentEntries = useMemo(() =>
    Object.entries(analysis?.stats?.agent_distribution ?? {}),
  [analysis]);
  const maxAgent = useMemo(() => Math.max(...agentEntries.map(([,c]) => c), 1), [agentEntries]);

  const monthEntries = useMemo(() =>
    Object.entries(analysis?.stats?.month_distribution ?? {}).sort(),
  [analysis]);
  const maxMonth = useMemo(() => Math.max(...monthEntries.map(([,c]) => c), 1), [monthEntries]);

  const propEntries = useMemo(() =>
    Object.entries(analysis?.stats?.properties ?? {}),
  [analysis]);

  // Top similarity pairs from graph
  const topPairs = useMemo(() => {
    if (!graphData?.edges) return [];
    return [...graphData.edges]
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, 6)
      .map(e => ({
        ...e,
        sourceText: nodeMap.get(e.source)?.text ?? e.source,
        targetText: nodeMap.get(e.target)?.text ?? e.target,
      }));
  }, [graphData, nodeMap]);

  if (loading) {
    return (
      <div className="flex h-screen bg-[#0b0b10] items-center justify-center">
        <div className="text-[#a1a1aa] text-sm animate-pulse">Loading dashboard…</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex h-screen bg-[#0b0b10] items-center justify-center flex-col gap-3">
        <p className="text-[#f87171] text-sm">{error}</p>
        <button onClick={fetchData} className="px-4 py-1.5 rounded-lg bg-[#1e1e2c] text-[#a1a1aa] text-xs hover:text-[#e4e4e7] transition">Retry</button>
      </div>
    );
  }

  return (
    <div className="flex h-screen bg-[#0b0b10] text-[#e4e4e7] font-sans overflow-hidden">
      <Sidebar page={page} setPage={setPage} stats={stats} coachingSummary={coachingSummary} />

      {/* Main Content */}
      <main className="flex-1 overflow-y-auto">

        {/* ============================================================
             GRAPH PAGE
             ============================================================ */}
        {page === "graph" && (
          <div className="relative w-full h-full">
            <DeckGL
              views={new OrbitView({ orbitAxis: "Y" })}
              viewState={viewState}
              controller={true}
              layers={graphLayers}
              onViewStateChange={({ viewState: vs }) => setViewState(vs as OrbitViewState)}
              getCursor={() => "grab"}
            />
            {/* Tooltip */}
            {tooltip && (
              <div className="fixed z-10 bg-[#0f0f18]/95 backdrop-blur-md rounded-xl px-4 py-2.5 text-xs text-[#e4e4e7] border border-[#1e1e2c] pointer-events-none max-w-md whitespace-pre-line shadow-xl"
                   style={{ left: tooltipPos.x, top: tooltipPos.y }}>
                {tooltip}
              </div>
            )}
            {/* Detail popup */}
            {selectedUtterance && (
              <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm" onClick={() => setSelectedUtterance(null)}>
                <div className="bg-[#0f0f18] rounded-xl border border-[#1e1e2c] p-5 max-w-lg w-full mx-4 shadow-2xl" onClick={e => e.stopPropagation()}>
                  <div className="flex justify-between items-start mb-3">
                    <div>
                      <span className="text-[11px] text-[#71717a] font-mono">{selectedUtterance.id.slice(0, 12)}…</span>
                      <span className="ml-2 px-2 py-0.5 rounded text-[11px] font-medium" style={{ background: agentHex(selectedUtterance.source_agent) + "22", color: agentHex(selectedUtterance.source_agent) }}>{selectedUtterance.source_agent}</span>
                      {selectedUtterance.timestamp && (
                        <span className="ml-2 text-[11px] text-[#71717a]">{new Date(selectedUtterance.timestamp).toLocaleDateString()}</span>
                      )}
                    </div>
                    <button onClick={() => setSelectedUtterance(null)} className="text-[#71717a] hover:text-[#e4e4e7]">&times;</button>
                  </div>
                  <p className="text-sm text-[#e4e4e7] leading-relaxed whitespace-pre-wrap break-words max-h-60 overflow-y-auto">{selectedUtterance.text}</p>
                  <div className="mt-3 pt-3 border-t border-[#1e1e2c] flex gap-4 text-[11px] text-[#71717a]">
                    <span>Turn #{selectedUtterance.turn_index}</span>
                    <span>Conv: {selectedUtterance.conversation_id.slice(0, 16)}…</span>
                  </div>
                </div>
              </div>
            )}
            {/* Graph legend overlay */}
            <div className="absolute top-4 right-4 z-10 bg-[#0f0f18]/90 backdrop-blur-md rounded-xl border border-[#1e1e2c] px-3 py-2.5 text-[11px] space-y-1">
              <p className="text-[#71717a] mb-1.5 font-semibold uppercase tracking-wider">Legend</p>
              {agentEntries.map(([agent, count]) => (
                <div key={agent} className="flex items-center gap-2">
                  <span className="w-2.5 h-2.5 rounded-full" style={{ background: agentHex(agent) }} />
                  <span className="text-[#a1a1aa]">{agent}</span>
                  <span className="text-[#71717a] font-mono ml-auto">{count}</span>
                </div>
              ))}
              <div className="border-t border-[#1e1e2c] mt-1.5 pt-1.5 text-[#71717a]">
                <span>Node size ∝ words</span>
                <br />
                <span>Edge width ∝ similarity</span>
              </div>
            </div>
            <div className="absolute bottom-4 left-4 z-10 text-[11px] text-[#71717a] bg-[#0f0f18]/80 backdrop-blur-md rounded-lg px-3 py-1.5 border border-[#1e1e2c]">
              {graphData?.nodes.length ?? 0} nodes · {graphData?.edges.length ?? 0} edges · Drag to rotate · Scroll to zoom
            </div>
          </div>
        )}

        {/* ============================================================
             DASHBOARD PAGE
             ============================================================ */}
        {page === "dashboard" && (
          <div className="p-6 space-y-5">
            <h2 className="text-lg font-semibold text-[#e4e4e7]">Analytics Dashboard</h2>

            {/* Row 1: Key metrics */}
            <div className="grid grid-cols-4 gap-4">
              <MetricCard label="Utterances" value={analysis?.stats.utterance_count ?? 0} color="#5eead4" />
              <MetricCard label="Conversations" value={analysis?.stats.conversation_count ?? 0} color="#818cf8" />
              <MetricCard label="Avg Words" value={analysis?.stats.avg_words?.toFixed(1) ?? "—"} color="#fbbf24" />
              <MetricCard label="Max Words" value={analysis?.stats.max_words ?? "—"} color="#fb923c" />
            </div>

            {/* Row 2: Agent + Time distributions */}
            <div className="grid grid-cols-2 gap-4">
              <Card title="Agent Distribution">
                <div className="space-y-2">
                  {agentEntries.map(([agent, count]) => (
                    <MiniBar key={agent} label={agent} value={count} max={maxAgent} color={agentHex(agent)} />
                  ))}
                </div>
              </Card>
              <Card title="Time Distribution">
                <div className="space-y-2">
                  {monthEntries.map(([month, count]) => (
                    <MiniBar key={month} label={month} value={count} max={maxMonth} color="#818cf8" />
                  ))}
                </div>
              </Card>
            </div>

            {/* Row 3: Properties + Top Similarity */}
            <div className="grid grid-cols-2 gap-4">
              <Card title="Detected Properties">
                <div className="flex flex-wrap gap-1.5">
                  {propEntries.map(([prop, count]) => (
                    <span key={prop} className="px-2.5 py-1 rounded-full text-[11px] bg-[#1e1e2c] text-[#a1a1aa] border border-[#2e2e3c]">
                      {prop.replace(/_/g, " ")} <span className="text-[#e4e4e7] font-mono ml-0.5">{count}</span>
                    </span>
                  ))}
                  {propEntries.length === 0 && <span className="text-[#71717a] text-xs">No properties detected</span>}
                </div>
              </Card>
              <Card title="Top Similarity Pairs">
                {topPairs.length > 0 ? (
                  <div className="space-y-2">
                    {topPairs.map((p, i) => (
                      <div key={i} className="text-[11px] flex items-start gap-2">
                        <span className="text-[#5eead4] font-mono shrink-0 w-10">{p.similarity.toFixed(3)}</span>
                        <span className="text-[#a1a1aa] truncate">
                          <span className="text-[#e4e4e7]">"{p.sourceText.slice(0, 40)}…"</span>
                          <span className="text-[#71717a] mx-1">↔</span>
                          <span className="text-[#e4e4e7]">"{p.targetText.slice(0, 40)}…"</span>
                        </span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-[#71717a] text-xs">Run `build-graph` to compute similarities</p>
                )}
              </Card>
            </div>

            {/* Row 4: Style Warnings */}
            {analysis?.stats?.style_warnings && analysis.stats.style_warnings.length > 0 && (
              <Card title={`Style Warnings (${analysis.stats.style_warnings.length})`}>
                <div className="space-y-2 max-h-64 overflow-y-auto">
                  {analysis.stats.style_warnings.slice(0, 20).map((w, i) => (
                    <div key={i} className="text-[11px] border-l-2 border-[#fbbf24]/40 pl-3 py-0.5">
                      <p className="text-[#fbbf24]">{w.message}</p>
                      <p className="text-[#71717a] mt-0.5 italic">"{w.excerpt}"</p>
                    </div>
                  ))}
                </div>
              </Card>
            )}
          </div>
        )}

        {/* ============================================================
             COACHING PAGE
             ============================================================ */}
        {page === "coaching" && (
          <div className="p-6 space-y-5">
            <h2 className="text-lg font-semibold text-[#e4e4e7]">AI Coaching Report</h2>

            {coachingSummary && coachingSummary.total_coached > 0 ? (
              <>
                {/* Summary cards */}
                <div className="grid grid-cols-4 gap-4">
                  <MetricCard label="Coached" value={coachingSummary.total_coached} color="#c084fc" />
                  <MetricCard label="Avg Clarity" value={`${coachingSummary.avg_clarity.toFixed(1)}/5`} color="#5eead4" />
                  <div className="col-span-2 bg-[#0f0f18] rounded-xl border border-[#1e1e2c] p-5">
                    <h3 className="text-xs font-semibold text-[#71717a] uppercase tracking-widest mb-3">Dominant Style</h3>
                    <p className="text-2xl font-semibold text-[#e4e4e7] capitalize">{coachingSummary.dominant_style.replace(/_/g, " ")}</p>
                  </div>
                </div>

                {/* Common issues + Top tips */}
                <div className="grid grid-cols-2 gap-4">
                  <Card title="Common Issues">
                    <ul className="space-y-1.5">
                      {coachingSummary.common_issues.map((issue, i) => (
                        <li key={i} className="text-xs text-[#a1a1aa] flex gap-2">
                          <span className="text-[#fbbf24] shrink-0">⚠</span>
                          <SafeMarkdown text={issue} />
                        </li>
                      ))}
                    </ul>
                  </Card>
                  <Card title="Top Tips">
                    <ul className="space-y-1.5">
                      {coachingSummary.top_tips.map((tip, i) => (
                        <li key={i} className="text-xs text-[#a1a1aa] flex gap-2">
                          <span className="text-[#5eead4] shrink-0">✦</span>
                          <SafeMarkdown text={tip} />
                        </li>
                      ))}
                    </ul>
                  </Card>
                </div>

                {/* Per-utterance coaching table */}
                <Card title={`All Coaching Entries (${coaching.length})`}>
                  <div className="space-y-2 max-h-[60vh] overflow-y-auto">
                    {coaching.map((c) => (
                      <div key={c.utterance_id}
                        onClick={() => setSelectedCoaching(c)}
                        className="flex items-center gap-3 p-2.5 rounded-lg hover:bg-[#14141c] cursor-pointer transition border border-transparent hover:border-[#1e1e2c]">
                        <span className="w-8 h-8 rounded-lg flex items-center justify-center text-xs font-bold" style={{ background: agentHex(c.source_agent) + "22", color: agentHex(c.source_agent) }}>
                          {c.clarity_score}
                        </span>
                        <div className="flex-1 min-w-0">
                          <p className="text-xs text-[#e4e4e7] truncate">{c.text.slice(0, 100)}</p>
                          <p className="text-[11px] text-[#71717a] capitalize mt-0.5">{c.source_agent} · {c.interaction_style.replace(/_/g, " ")}</p>
                        </div>
                        <span className="text-[11px] text-[#71717a] shrink-0">{c.clarity_score}/5</span>
                      </div>
                    ))}
                  </div>
                </Card>
              </>
            ) : (
              <Card>
                <p className="text-sm text-[#a1a1aa]">
                  No coaching data yet. Run <code className="px-1.5 py-0.5 rounded bg-[#1e1e2c] text-[#5eead4] text-xs">agentrace-cli analyze --coach</code> with your DEEPSEEK_API_KEY set.
                </p>
              </Card>
            )}
          </div>
        )}
      </main>

      {/* Coaching detail modal */}
      {selectedCoaching && <CoachingDetail item={selectedCoaching} onClose={() => setSelectedCoaching(null)} />}
    </div>
  );
}

// ======================================================================
// Sub-components
// ======================================================================

function MetricCard({ label, value, color }: { label: string; value: string | number; color: string }) {
  return (
    <div className="bg-[#0f0f18] rounded-xl border border-[#1e1e2c] p-5">
      <p className="text-[11px] font-semibold text-[#71717a] uppercase tracking-widest mb-1">{label}</p>
      <p className="text-2xl font-bold tabular-nums" style={{ color }}>{value}</p>
    </div>
  );
}
