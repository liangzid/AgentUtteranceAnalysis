import { DeckGL } from "@deck.gl/react";
import { ScatterplotLayer } from "@deck.gl/layers";
import { MapView } from "@deck.gl/core";
import type { MapViewState } from "@deck.gl/core";
import { useState, useMemo } from "react";

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

function generateMockPoints(count: number) {
  return Array.from({ length: count }, (_, i) => ({
    position: [
      (Math.random() - 0.5) * 200,
      (Math.random() - 0.5) * 200,
      (Math.random() - 0.5) * 100,
    ] as [number, number, number],
    size: Math.random() * 8 + 2,
    color: [
      Math.random() * 200 + 55,
      Math.random() * 100 + 100,
      Math.random() * 150 + 105,
    ] as [number, number, number],
    id: `point-${i}`,
    label: `Utterance #${i}`,
  }));
}

export default function App() {
  const [viewState, setViewState] = useState<MapViewState>(INITIAL_VIEW_STATE);
  const points = useMemo(() => generateMockPoints(500), []);

  const layers = [
    new ScatterplotLayer({
      id: "utterance-points",
      data: points,
      getPosition: (d) => d.position,
      getRadius: (d) => d.size,
      getFillColor: (d) => d.color,
      radiusMinPixels: 1,
      radiusMaxPixels: 30,
      pickable: true,
      opacity: 0.8,
      onHover: (info: { object?: { label?: string }; x: number; y: number }) => {
        // Tooltip will be added later
        if (info.object) {
          document.body.style.cursor = "pointer";
        } else {
          document.body.style.cursor = "grab";
        }
      },
    }),
  ];

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
      {/* Info overlay */}
      <div className="absolute top-4 left-4 z-10 bg-black/60 backdrop-blur-md rounded-lg px-4 py-3 text-sm border border-white/10">
        <h1 className="text-lg font-bold text-white mb-1">Agentrace</h1>
        <p className="text-gray-400 text-xs">
          {points.length.toLocaleString()} utterances &middot; Drag to rotate
          &middot; Scroll to zoom
        </p>
      </div>
    </div>
  );
}
