import { DeckGL } from "@deck.gl/react";
import { ScatterplotLayer, LineLayer } from "@deck.gl/layers";
import { MapView } from "@deck.gl/core";
import type { MapViewState } from "@deck.gl/core";
import { useState, useMemo, useEffect, useCallback } from "react";

const INITIAL_VIEW_STATE: MapViewState = {
  longitude: 0, latitude: 0, zoom: 3, pitch: 45, bearing: 0,
  maxZoom: 20, minZoom: 0.5, maxPitch: 89, minPitch: 0,
};

// --- Types ---

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
  };
}

interface UtteranceItem {
  id: string;
  source_agent: string;
  conversation_id: string;
  turn_index: number;
  timestamp: string | null;
  text: string;
  source_path: string;
}

interface UtterancesResponse { utterances: UtteranceItem[] }

type ColorMode = "agent" | "time" | "properties";

// --- Constants ---

const AGENT_COLORS: Record<string, [number, number, number]> = {
  codex: [100, 200, 255],
  "claude-code": [255, 180, 100],
  opencode: [100, 255, 150],
  openclaw: [255, 100, 200],
  "kilo-code": [200, 150, 255],
};
const DEFAULT_COLOR: [number, number, number] = [150, 150, 200];

function agentColor(agent: string): [number, number, number] {
  return AGENT_COLORS[agent] ?? DEFAULT_COLOR;
}

function timeColor(ts: string | null): [number, number, number] {
  if (!ts) return [100, 100, 100];
  const d = new Date(ts);
  const year = d.getFullYear();
  const days = (d.getTime() - new Date(year, 0, 0).getTime()) / 86400000;
  const hue = (days / 365) * 300 + 200;
  const rgb = hslToRgb(hue % 360, 0.7, 0.6);
  return rgb;
}

function propColor(props: Record<string, number> | undefined): [number, number, number] {
  if (!props) return [100, 100, 140];
  if (props.contains_code_block) return [100, 255, 200];
  if (props.question_or_inquiry) return [255, 200, 100];
  if (props.request_or_instruction) return [100, 180, 255];
  if (props.long_context_prompt) return [255, 100, 100];
  return [160, 160, 200];
}

function hslToRgb(h: number, s: number, l: number): [number, number, number] {
  const c = (1 - Math.abs(2 * l - 1)) * s;
  const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
  const m = l - c / 2;
  let [r, g, b] = [0, 0, 0];
  if (h < 60) [r, g, b] = [c, x, 0];
  else if (h < 120) [r, g, b] = [x, c, 0];
  else if (h < 180) [r, g, b] = [0, c, x];
  else if (h < 240) [r, g, b] = [0, x, c];
  else if (h < 300) [r, g, b] = [x, 0, c];
  else [r, g, b] = [c, 0, x];
  return [Math.round((r + m) * 255), Math.round((g + m) * 255), Math.round((b + m) * 255)];
}

// --- 3D Point Generation ---

function spiral3D(i: number, total: number, spread: number): [number, number, number] {
  const t = i / Math.max(total, 1);
  const r = 10 + t * spread;
  const theta = t * Math.PI * 8;
  const phi = Math.acos(1 - 2 * t);
  return [
    r * Math.sin(phi) * Math.cos(theta),
    r * Math.sin(phi) * Math.sin(theta),
    r * Math.cos(phi) - spread * 0.3,
  ];
}

// --- Detail Popup ---

function DetailPopup({ item, onClose }: { item: UtteranceItem; onClose: () => void }) {
  return (
    <div className="absolute inset-0 z-20 flex items-center justify-center pointer-events-none">
      <div className="pointer-events-auto bg-[#12121a]/95 backdrop-blur-xl rounded-xl border border-white/15 p-6 max-w-lg w-full mx-4 shadow-2xl">
        <div className="flex justify-between items-start mb-4">
          <div>
            <span className="text-xs text-gray-500 font-mono">{item.id.slice(0, 12)}…</span>
            <span className="ml-2 px-2 py-0.5 rounded text-xs bg-white/10 text-gray-300">{item.source_agent}</span>
            {item.timestamp && (
              <span className="ml-2 text-xs text-gray-500">{new Date(item.timestamp).toLocaleDateString()}</span>
            )}
          </div>
          <button onClick={onClose} className="text-gray-500 hover:text-white text-lg leading-none">&times;</button>
        </div>
        <p className="text-sm text-gray-200 leading-relaxed whitespace-pre-wrap break-words max-h-60 overflow-y-auto">
          {item.text}
        </p>
        <div className="mt-3 pt-3 border-t border-white/10 flex gap-4 text-xs text-gray-500">
          <span>Turn #{item.turn_index}</span>
          <span>Conv: {item.conversation_id.slice(0, 16)}…</span>
        </div>
      </div>
    </div>
  );
}

// --- Main App ---

export default function App() {
  const [viewState, setViewState] = useState<MapViewState>(INITIAL_VIEW_STATE);
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [analysis, setAnalysis] = useState<AnalysisResponse | null>(null);
  const [utterances, setUtterances] = useState<UtteranceItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [colorMode, setColorMode] = useState<ColorMode>("agent");
  const [tooltip, setTooltip] = useState<string | null>(null);
  const [selected, setSelected] = useState<UtteranceItem | null>(null);
  const [timeRange, setTimeRange] = useState<[number, number] | null>(null);

  const fetchData = useCallback(async () => {
    try {
      const [sRes, aRes, uRes] = await Promise.all([
        fetch("/api/v1/stats"), fetch("/api/v1/analysis"), fetch("/api/v1/utterances"),
      ]);
      if (sRes.ok) setStats(await sRes.json());
      if (aRes.ok) setAnalysis(await aRes.json());
      if (uRes.ok) {
        const data: UtterancesResponse = await uRes.json();
        setUtterances(data.utterances);
      }
      setError(null);
    } catch { setError("Cannot reach API. Start with: agentrace serve"); }
    finally { setLoading(false); }
  }, []);

  useEffect(() => { fetchData(); }, [fetchData]);

  // Time range from data
  const timeBounds = useMemo(() => {
    const times = utterances.map(u => u.timestamp ? new Date(u.timestamp).getTime() : NaN).filter(t => !isNaN(t));
    if (times.length === 0) return null;
    return [Math.min(...times), Math.max(...times)] as [number, number];
  }, [utterances]);

  const effectiveRange = timeRange ?? timeBounds ?? [0, 1];

  // Filtered utterances
  const filtered = useMemo(() => {
    if (!timeRange || !timeBounds || (timeRange[0] === timeBounds[0] && timeRange[1] === timeBounds[1])) {
      return utterances;
    }
    return utterances.filter(u => {
      if (!u.timestamp) return true;
      const t = new Date(u.timestamp).getTime();
      return t >= effectiveRange[0] && t <= effectiveRange[1];
    });
  }, [utterances, timeRange, timeBounds, effectiveRange]);

  // Build analysis properties map per utterance
  const propMap = useMemo(() => {
    const m = new Map<string, Record<string, number>>();
    if (!analysis?.stats) return m;
    // All utterances share global properties for now
    const props = analysis.stats.properties;
    for (const u of utterances) m.set(u.id, props);
    return m;
  }, [analysis, utterances]);

  // Points with 3D spiral layout
  const points = useMemo(() => {
    if (filtered.length === 0) return [];
    return filtered.map((u, i) => {
      const pos = spiral3D(i, filtered.length, 80);
      const colorFn = colorMode === "time" ? () => timeColor(u.timestamp)
        : colorMode === "properties" ? () => propColor(propMap.get(u.id))
        : () => agentColor(u.source_agent);
      return { position: pos, size: 3 + Math.random() * 4, color: colorFn(), id: u.id, utterance: u };
    });
  }, [filtered, colorMode, propMap]);

  // Trajectory lines: connect utterances in the same conversation, ordered by turn_index
  const trajectoryLines = useMemo(() => {
    const convMap = new Map<string, typeof points>();
    for (const p of points) {
      const cid = p.utterance.conversation_id;
      if (!convMap.has(cid)) convMap.set(cid, []);
      convMap.get(cid)!.push(p);
    }
    const lines: { sourcePosition: [number,number,number]; targetPosition: [number,number,number]; color: [number,number,number] }[] = [];
    for (const [, pts] of convMap) {
      if (pts.length < 2) continue;
      pts.sort((a, b) => a.utterance.turn_index - b.utterance.turn_index);
      for (let i = 0; i < pts.length - 1; i++) {
        lines.push({
          sourcePosition: pts[i].position,
          targetPosition: pts[i + 1].position,
          color: agentColor(pts[i].utterance.source_agent).map(c => c * 0.6) as [number, number, number],
        });
      }
    }
    return lines;
  }, [points]);

  // Layers
  const layers = useMemo(() => {
    const ls: any[] = [];
    if (trajectoryLines.length > 0) {
      ls.push(new LineLayer({
        id: "trajectory-lines", data: trajectoryLines,
        getSourcePosition: (d: any) => d.sourcePosition,
        getTargetPosition: (d: any) => d.targetPosition,
        getColor: (d: any) => d.color,
        opacity: 0.25, widthMinPixels: 0.5, widthMaxPixels: 1.5,
      }));
    }
    ls.push(new ScatterplotLayer({
      id: "utterance-points", data: points,
      getPosition: (d: any) => d.position,
      getRadius: (d: any) => d.size,
      getFillColor: (d: any) => d.color,
      radiusMinPixels: 1.5, radiusMaxPixels: 28,
      pickable: true, opacity: 0.9,
      onClick: (info: any) => {
        if (info.object?.utterance) setSelected(info.object.utterance);
      },
      onHover: (info: any) => {
        if (info.object?.utterance) {
          const u = info.object.utterance as UtteranceItem;
          setTooltip(`${u.source_agent} · turn ${u.turn_index} · "${u.text.slice(0, 80)}${u.text.length > 80 ? "…" : ""}"`);
          document.body.style.cursor = "pointer";
        } else { setTooltip(null); document.body.style.cursor = "grab"; }
      },
    }));
    return ls;
  }, [points, trajectoryLines]);

  const agentEntries = Object.entries(analysis?.stats?.agent_distribution ?? {});
  const modes: { key: ColorMode; label: string }[] = [
    { key: "agent", label: "Agent" }, { key: "time", label: "Time" }, { key: "properties", label: "Props" },
  ];

  return (
    <div className="relative w-full h-full bg-[#0a0a0f] font-sans">
      <DeckGL
        initialViewState={viewState} controller={true}
        views={new MapView({ repeat: true })} layers={layers}
        onViewStateChange={({ viewState: vs }) => setViewState(vs as MapViewState)}
        getCursor={() => "grab"}
      />

      {/* Tooltip */}
      {tooltip && (
        <div className="absolute bottom-8 left-1/2 -translate-x-1/2 z-10 bg-black/85 backdrop-blur-md rounded-lg px-4 py-2 text-xs text-white border border-white/10 pointer-events-none max-w-lg truncate">
          {tooltip}
        </div>
      )}

      {/* Detail popup */}
      {selected && <DetailPopup item={selected} onClose={() => setSelected(null)} />}

      {/* Top-left: stats */}
      <div className="absolute top-4 left-4 z-10 bg-black/65 backdrop-blur-md rounded-lg px-3 py-2.5 text-xs border border-white/10 min-w-[200px]">
        <h1 className="text-sm font-bold text-white mb-1">Agentrace</h1>
        {loading ? <p className="text-gray-400 animate-pulse">Connecting…</p>
          : error ? <p className="text-red-400">{error}</p> : <>
            <p className="text-gray-500 mb-2">{filtered.length} of {utterances.length} pts · Drag · Scroll · Click</p>
            <div className="space-y-0.5">
              <Row label="Utterances" value={stats?.utterances ?? 0} />
              <Row label="Conversations" value={stats?.conversations ?? 0} />
              {analysis && <>
                <Row label="Avg Words" value={analysis.stats.avg_words.toFixed(1)} />
                <Row label="Max Words" value={analysis.stats.max_words} />
              </>}
            </div>
          </>}
      </div>

      {/* Top-right: color mode + agent legend */}
      <div className="absolute top-4 right-4 z-10 flex flex-col gap-2">
        <div className="bg-black/65 backdrop-blur-md rounded-lg px-2 py-1.5 border border-white/10 flex gap-1">
          {modes.map(m => (
            <button key={m.key}
              onClick={() => setColorMode(m.key)}
              className={`px-2 py-0.5 rounded text-xs transition ${colorMode === m.key ? "bg-white/20 text-white" : "text-gray-500 hover:text-gray-300"}`}>
              {m.label}
            </button>
          ))}
        </div>
        {agentEntries.length > 0 && (
          <div className="bg-black/65 backdrop-blur-md rounded-lg px-2.5 py-2 text-xs border border-white/10">
            {agentEntries.map(([agent, count]) => (
              <div key={agent} className="flex items-center gap-1.5 mb-0.5">
                <span className="inline-block w-2 h-2 rounded-full" style={{ background: `rgb(${agentColor(agent).join(",")})` }} />
                <span className="text-gray-300">{agent}</span>
                <span className="text-gray-500 ml-auto font-mono">{count}</span>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Bottom-left: properties */}
      {analysis?.stats?.properties && Object.keys(analysis.stats.properties).length > 0 && (
        <div className="absolute bottom-4 left-4 z-10 bg-black/65 backdrop-blur-md rounded-lg px-2.5 py-2 text-xs border border-white/10">
          <p className="text-gray-500 mb-1">Properties</p>
          {Object.entries(analysis.stats.properties).map(([prop, count]) => (
            <div key={prop} className="flex justify-between gap-3 text-xs">
              <span className="text-gray-300">{prop.replace(/_/g, " ")}</span>
              <span className="text-white font-mono">{count}</span>
            </div>
          ))}
        </div>
      )}

      {/* Bottom: time slider */}
      {timeBounds && timeBounds[0] < timeBounds[1] && (
        <div className="absolute bottom-4 left-1/2 -translate-x-1/2 z-10 bg-black/65 backdrop-blur-md rounded-lg px-3 py-2 border border-white/10 flex items-center gap-3 text-xs">
          <span className="text-gray-500 w-20 text-right">{new Date(effectiveRange[0]).toLocaleDateString()}</span>
          <input type="range" min={timeBounds[0]} max={timeBounds[1]} step={86400000}
            value={effectiveRange[0]} className="w-40 h-1 accent-cyan-400"
            onChange={e => setTimeRange([+e.target.value, effectiveRange[1]])} />
          <input type="range" min={timeBounds[0]} max={timeBounds[1]} step={86400000}
            value={effectiveRange[1]} className="w-40 h-1 accent-cyan-400"
            onChange={e => setTimeRange([effectiveRange[0], +e.target.value])} />
          <span className="text-gray-500 w-20">{new Date(effectiveRange[1]).toLocaleDateString()}</span>
          {timeRange && (
            <button onClick={() => setTimeRange(null)} className="text-gray-500 hover:text-white px-1">↺</button>
          )}
        </div>
      )}
    </div>
  );
}

function Row({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="flex justify-between text-xs">
      <span className="text-gray-400">{label}</span>
      <span className="text-white font-mono">{value}</span>
    </div>
  );
}
