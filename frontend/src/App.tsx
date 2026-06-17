import { DeckGL } from "@deck.gl/react";
import { ScatterplotLayer } from "@deck.gl/layers";
import { MapView } from "@deck.gl/core";
import type { MapViewState } from "@deck.gl/core";
import { useState, useMemo, useEffect, useCallback } from "react";

const INITIAL_VIEW_STATE: MapViewState = {
  longitude: 0,
  latitude: 0,
  zoom: 3,
  pitch: 45,
  bearing: 0,
  maxZoom: 20,
  minZoom: 0.5,
  maxPitch: 89,
  minPitch: 0,
};

interface StatsResponse {
  utterances: number;
  conversations: number;
  agents: { agent: string; count: number }[];
  months: { month: string; count: number }[];
}

interface AnalysisResponse {
  stats: {
    utterance_count: number;
    conversation_count: number;
    avg_words: number;
    median_words: number;
    max_words: number;
    avg_chars: number;
    agent_distribution: Record<string, number>;
    month_distribution: Record<string, number>;
    properties: Record<string, number>;
  };
}

interface PointData {
  position: [number, number, number];
  size: number;
  color: [number, number, number];
  id: string;
  label: string;
}

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

function generateCloudPoints(
  count: number,
  agents: { agent: string; count: number }[]
): PointData[] {
  if (count === 0) return [];
  return Array.from({ length: Math.min(count, 2000) }, (_, i) => {
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos(2 * Math.random() - 1);
    const r = 30 + Math.random() * 70;
    const x = r * Math.sin(phi) * Math.cos(theta);
    const y = r * Math.sin(phi) * Math.sin(theta);
    const z = r * Math.cos(phi);

    // Pick a random agent for coloring (weighted by count)
    const totalAgentCount = agents.reduce((s, a) => s + a.count, 0);
    let roll = Math.random() * totalAgentCount;
    let chosenAgent = agents[0]?.agent ?? "unknown";
    for (const a of agents) {
      roll -= a.count;
      if (roll <= 0) {
        chosenAgent = a.agent;
        break;
      }
    }

    return {
      position: [x, y, z] as [number, number, number],
      size: 2 + Math.random() * 6,
      color: agentColor(chosenAgent),
      id: `u-${i}`,
      label: `Utterance #${i} · ${chosenAgent}`,
    };
  });
}

export default function App() {
  const [viewState, setViewState] = useState<MapViewState>(INITIAL_VIEW_STATE);
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [analysis, setAnalysis] = useState<AnalysisResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [tooltip, setTooltip] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    try {
      const [statsRes, analysisRes] = await Promise.all([
        fetch("/api/v1/stats"),
        fetch("/api/v1/analysis"),
      ]);
      if (statsRes.ok) setStats(await statsRes.json());
      if (analysisRes.ok) setAnalysis(await analysisRes.json());
      setError(null);
    } catch {
      setError("Cannot reach API. Start with: agentrace serve");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const points = useMemo(
    () => generateCloudPoints(stats?.utterances ?? 0, stats?.agents ?? []),
    [stats]
  );

  const layers = [
    new ScatterplotLayer({
      id: "utterance-points",
      data: points,
      getPosition: (d: PointData) => d.position,
      getRadius: (d: PointData) => d.size,
      getFillColor: (d: PointData) => d.color,
      radiusMinPixels: 1,
      radiusMaxPixels: 30,
      pickable: true,
      opacity: 0.85,
      onHover: (info: { object?: PointData; x: number; y: number }) => {
        if (info.object) {
          setTooltip(info.object.label);
          document.body.style.cursor = "pointer";
        } else {
          setTooltip(null);
          document.body.style.cursor = "grab";
        }
      },
    }),
  ];

  const agentEntries = Object.entries(analysis?.stats?.agent_distribution ?? {});

  return (
    <div className="relative w-full h-full bg-[#0a0a0f]">
      <DeckGL
        initialViewState={viewState}
        controller={true}
        views={new MapView({ repeat: true })}
        layers={layers}
        onViewStateChange={({ viewState: vs }) =>
          setViewState(vs as MapViewState)
        }
        getCursor={() => "grab"}
      />

      {/* Tooltip */}
      {tooltip && (
        <div className="absolute bottom-8 left-1/2 -translate-x-1/2 z-10 bg-black/80 backdrop-blur-md rounded-lg px-4 py-2 text-sm text-white border border-white/10 pointer-events-none">
          {tooltip}
        </div>
      )}

      {/* Stats panel — top left */}
      <div className="absolute top-4 left-4 z-10 bg-black/60 backdrop-blur-md rounded-lg px-4 py-3 text-sm border border-white/10 min-w-[220px]">
        <h1 className="text-lg font-bold text-white mb-1">Agentrace</h1>
        {loading ? (
          <p className="text-gray-400 text-xs animate-pulse">Connecting...</p>
        ) : error ? (
          <p className="text-red-400 text-xs">{error}</p>
        ) : (
          <>
            <p className="text-gray-400 text-xs mb-2">
              {points.length.toLocaleString()} points · Drag · Scroll · Pitch
            </p>
            <div className="space-y-1">
              <div className="flex justify-between text-xs">
                <span className="text-gray-400">Utterances</span>
                <span className="text-white font-mono">
                  {stats?.utterances ?? 0}
                </span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-gray-400">Conversations</span>
                <span className="text-white font-mono">
                  {stats?.conversations ?? 0}
                </span>
              </div>
              {analysis && (
                <>
                  <div className="flex justify-between text-xs">
                    <span className="text-gray-400">Avg Words</span>
                    <span className="text-white font-mono">
                      {analysis.stats.avg_words.toFixed(1)}
                    </span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span className="text-gray-400">Max Words</span>
                    <span className="text-white font-mono">
                      {analysis.stats.max_words}
                    </span>
                  </div>
                </>
              )}
            </div>
          </>
        )}
      </div>

      {/* Agent legend — top right */}
      {agentEntries.length > 0 && (
        <div className="absolute top-4 right-4 z-10 bg-black/60 backdrop-blur-md rounded-lg px-3 py-2 text-xs border border-white/10">
          <p className="text-gray-400 mb-1.5">Agents</p>
          {agentEntries.map(([agent, count]) => (
            <div key={agent} className="flex items-center gap-2 mb-1">
              <span
                className="inline-block w-2.5 h-2.5 rounded-full"
                style={{
                  background: `rgb(${agentColor(agent).join(",")})`,
                }}
              />
              <span className="text-gray-300">{agent}</span>
              <span className="text-gray-500 ml-auto font-mono">{count}</span>
            </div>
          ))}
        </div>
      )}

      {/* Properties panel — bottom left */}
      {analysis?.stats?.properties &&
        Object.keys(analysis.stats.properties).length > 0 && (
          <div className="absolute bottom-4 left-4 z-10 bg-black/60 backdrop-blur-md rounded-lg px-3 py-2 text-xs border border-white/10">
            <p className="text-gray-400 mb-1">Properties</p>
            {Object.entries(analysis.stats.properties).map(([prop, count]) => (
              <div key={prop} className="flex justify-between gap-4 text-xs">
                <span className="text-gray-300">
                  {prop.replace(/_/g, " ")}
                </span>
                <span className="text-white font-mono">{count}</span>
              </div>
            ))}
          </div>
        )}
    </div>
  );
}
