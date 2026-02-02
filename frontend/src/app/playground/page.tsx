"use client";

import { useState, useCallback, useRef, useEffect, useMemo } from "react";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import {
  faAtom,
  faPlay,
  faSpinner,
  faCircleNodes,
  faArrowLeft,
  faTriangleExclamation,
  faMagnifyingGlassPlus,
  faMagnifyingGlassMinus,
  faExpand,
} from "@fortawesome/free-solid-svg-icons";
import { faGithub } from "@fortawesome/free-brands-svg-icons";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Slider } from "@/components/ui/slider";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import Link from "next/link";

// Types
interface Node {
  id: number;
  degree: number;
  clustering: number;
  importance: number;
}

interface Edge {
  source: number;
  target: number;
}

interface GraphData {
  nodes: Node[];
  edges: Edge[];
  n_nodes: number;
  n_edges: number;
  density: number;
  avg_clustering: number;
  avg_degree: number;
}

interface PredictionResult {
  gamma: number[];
  beta: number[];
  p_layers: number;
  graph: GraphData;
}

// Graph types with descriptions
const GRAPH_TYPES = [
  {
    value: "erdos_renyi",
    label: "Erdos-Renyi",
    description: "Graphe aleatoire uniforme",
  },
  {
    value: "barabasi_albert",
    label: "Barabasi-Albert",
    description: "Reseau scale-free",
  },
  {
    value: "watts_strogatz",
    label: "Watts-Strogatz",
    description: "Petit monde",
  },
  {
    value: "regular",
    label: "Regulier",
    description: "Degre constant",
  },
];

// Force-directed layout calculation (pure function, no hooks)
function calculateForceLayout(
  nodes: Node[],
  edges: Edge[],
  width: number,
  height: number
): { x: number; y: number }[] {
  if (nodes.length === 0) return [];

  // Use seeded random for consistent layout
  const seededRandom = (seed: number) => {
    const x = Math.sin(seed) * 10000;
    return x - Math.floor(x);
  };

  // Initialize positions
  const pos = nodes.map((_, i) => ({
    x: width / 2 + (seededRandom(i * 13) - 0.5) * width * 0.8,
    y: height / 2 + (seededRandom(i * 17 + 100) - 0.5) * height * 0.8,
  }));

  // Simple force simulation
  const iterations = 100;
  const k = Math.sqrt((width * height) / nodes.length) * 0.8;

  for (let iter = 0; iter < iterations; iter++) {
    const temp = 1 - iter / iterations;

    // Repulsion between all nodes
    for (let i = 0; i < nodes.length; i++) {
      let fx = 0;
      let fy = 0;

      for (let j = 0; j < nodes.length; j++) {
        if (i === j) continue;

        const dx = pos[i].x - pos[j].x;
        const dy = pos[i].y - pos[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy) || 1;

        const force = (k * k) / dist;
        fx += (dx / dist) * force;
        fy += (dy / dist) * force;
      }

      pos[i].x += fx * temp * 0.1;
      pos[i].y += fy * temp * 0.1;
    }

    // Attraction along edges
    for (const edge of edges) {
      const dx = pos[edge.target].x - pos[edge.source].x;
      const dy = pos[edge.target].y - pos[edge.source].y;
      const dist = Math.sqrt(dx * dx + dy * dy) || 1;

      const force = (dist * dist) / k;
      const fx = (dx / dist) * force * 0.05;
      const fy = (dy / dist) * force * 0.05;

      pos[edge.source].x += fx * temp;
      pos[edge.source].y += fy * temp;
      pos[edge.target].x -= fx * temp;
      pos[edge.target].y -= fy * temp;
    }

    // Keep in bounds
    const margin = 40;
    for (let i = 0; i < nodes.length; i++) {
      pos[i].x = Math.max(margin, Math.min(width - margin, pos[i].x));
      pos[i].y = Math.max(margin, Math.min(height - margin, pos[i].y));
    }
  }

  return pos;
}

// Graph Visualization Component
function GraphVisualization({
  graph,
  selectedNode,
  onNodeSelect,
}: {
  graph: GraphData | null;
  selectedNode: number | null;
  onNodeSelect: (id: number | null) => void;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const [dimensions, setDimensions] = useState({ width: 500, height: 400 });

  // Pan and zoom state
  const [viewBox, setViewBox] = useState({ x: 0, y: 0, width: 500, height: 400 });
  const [isPanning, setIsPanning] = useState(false);
  const [hasPanned, setHasPanned] = useState(false);
  const [startPan, setStartPan] = useState({ x: 0, y: 0 });
  const [zoom, setZoom] = useState(1);

  useEffect(() => {
    if (containerRef.current) {
      const rect = containerRef.current.getBoundingClientRect();
      setDimensions({ width: rect.width, height: 400 });
      setViewBox({ x: 0, y: 0, width: rect.width, height: 400 });
    }
  }, []);

  // Reset view when graph changes
  useEffect(() => {
    if (graph) {
      setViewBox({ x: 0, y: 0, width: dimensions.width, height: dimensions.height });
      setZoom(1);
    }
  }, [graph, dimensions.width, dimensions.height]);

  // Handle mouse wheel for zoom
  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();

    const scaleAmount = e.deltaY > 0 ? 1.1 : 0.9;
    const newZoom = Math.min(Math.max(zoom * scaleAmount, 0.2), 5);

    // Get mouse position relative to SVG
    const svg = svgRef.current;
    if (!svg) return;

    const rect = svg.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    // Convert mouse position to viewBox coordinates
    const viewBoxMouseX = viewBox.x + (mouseX / dimensions.width) * viewBox.width;
    const viewBoxMouseY = viewBox.y + (mouseY / dimensions.height) * viewBox.height;

    // Calculate new viewBox dimensions
    const newWidth = dimensions.width * (1 / newZoom);
    const newHeight = dimensions.height * (1 / newZoom);

    // Adjust viewBox to zoom towards mouse position
    const newX = viewBoxMouseX - (mouseX / dimensions.width) * newWidth;
    const newY = viewBoxMouseY - (mouseY / dimensions.height) * newHeight;

    setViewBox({ x: newX, y: newY, width: newWidth, height: newHeight });
    setZoom(newZoom);
  }, [zoom, viewBox, dimensions]);

  // Handle mouse down for pan start
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (e.button === 0) { // Left click
      setIsPanning(true);
      setHasPanned(false);
      setStartPan({ x: e.clientX, y: e.clientY });
    }
  }, []);

  // Handle mouse move for panning
  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (!isPanning) return;

    const dx = (e.clientX - startPan.x) * (viewBox.width / dimensions.width);
    const dy = (e.clientY - startPan.y) * (viewBox.height / dimensions.height);

    // Only count as panning if moved more than 3 pixels
    if (Math.abs(e.clientX - startPan.x) > 3 || Math.abs(e.clientY - startPan.y) > 3) {
      setHasPanned(true);
    }

    setViewBox(prev => ({
      ...prev,
      x: prev.x - dx,
      y: prev.y - dy,
    }));

    setStartPan({ x: e.clientX, y: e.clientY });
  }, [isPanning, startPan, viewBox, dimensions]);

  // Handle mouse up for pan end
  const handleMouseUp = useCallback(() => {
    setIsPanning(false);
    // Reset hasPanned after a short delay to allow click events to check it
    setTimeout(() => setHasPanned(false), 100);
  }, []);

  // Handle mouse leave
  const handleMouseLeave = useCallback(() => {
    setIsPanning(false);
    setHasPanned(false);
  }, []);

  // Reset view function
  const resetView = useCallback(() => {
    setViewBox({ x: 0, y: 0, width: dimensions.width, height: dimensions.height });
    setZoom(1);
  }, [dimensions]);

  // Calculate positions using useMemo to avoid recalculation on every render
  const positions = useMemo(() => {
    if (!graph) return [];
    return calculateForceLayout(
      graph.nodes,
      graph.edges,
      dimensions.width,
      dimensions.height
    );
  }, [graph, dimensions.width, dimensions.height]);

  if (!graph) {
    return (
      <div
        ref={containerRef}
        className="flex h-[400px] items-center justify-center rounded-lg border border-dashed border-border bg-secondary/30"
      >
        <div className="text-center text-muted-foreground">
          <FontAwesomeIcon icon={faCircleNodes} className="mb-2 h-8 w-8 opacity-50" />
          <p>Generez un graphe pour commencer</p>
        </div>
      </div>
    );
  }

  // Color scale for importance (blue to red)
  const getNodeColor = (importance: number) => {
    const r = Math.round(importance * 200 + 55);
    const g = Math.round((1 - importance) * 100 + 80);
    const b = Math.round((1 - importance) * 200 + 55);
    return `rgb(${r}, ${g}, ${b})`;
  };

  return (
    <div ref={containerRef} className="relative">
      {/* Zoom controls */}
      <div className="absolute right-2 top-2 z-10 flex flex-col gap-1">
        <button
          onClick={() => {
            const newZoom = Math.min(zoom * 1.3, 5);
            const centerX = viewBox.x + viewBox.width / 2;
            const centerY = viewBox.y + viewBox.height / 2;
            const newWidth = dimensions.width * (1 / newZoom);
            const newHeight = dimensions.height * (1 / newZoom);
            setViewBox({
              x: centerX - newWidth / 2,
              y: centerY - newHeight / 2,
              width: newWidth,
              height: newHeight,
            });
            setZoom(newZoom);
          }}
          className="flex h-8 w-8 items-center justify-center rounded bg-card border border-border text-sm font-medium hover:bg-secondary transition-colors"
          title="Zoom avant"
        >
          <FontAwesomeIcon icon={faMagnifyingGlassPlus} className="h-3.5 w-3.5" />
        </button>
        <button
          onClick={() => {
            const newZoom = Math.max(zoom * 0.7, 0.2);
            const centerX = viewBox.x + viewBox.width / 2;
            const centerY = viewBox.y + viewBox.height / 2;
            const newWidth = dimensions.width * (1 / newZoom);
            const newHeight = dimensions.height * (1 / newZoom);
            setViewBox({
              x: centerX - newWidth / 2,
              y: centerY - newHeight / 2,
              width: newWidth,
              height: newHeight,
            });
            setZoom(newZoom);
          }}
          className="flex h-8 w-8 items-center justify-center rounded bg-card border border-border text-sm font-medium hover:bg-secondary transition-colors"
          title="Zoom arriere"
        >
          <FontAwesomeIcon icon={faMagnifyingGlassMinus} className="h-3.5 w-3.5" />
        </button>
        <button
          onClick={resetView}
          className="flex h-8 w-8 items-center justify-center rounded bg-card border border-border text-xs font-medium hover:bg-secondary transition-colors"
          title="Reinitialiser la vue"
        >
          <FontAwesomeIcon icon={faExpand} className="h-3.5 w-3.5" />
        </button>
      </div>

      {/* Zoom indicator */}
      <div className="absolute left-2 top-2 z-10 rounded bg-card/90 px-2 py-1 text-xs text-muted-foreground">
        {Math.round(zoom * 100)}%
      </div>

      <svg
        ref={svgRef}
        width={dimensions.width}
        height={dimensions.height}
        viewBox={`${viewBox.x} ${viewBox.y} ${viewBox.width} ${viewBox.height}`}
        className={`rounded-lg border border-border bg-card ${isPanning ? "cursor-grabbing" : "cursor-grab"}`}
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseLeave}
      >
        {/* Edges */}
        {graph.edges.map((edge, i) => {
          const sourcePos = positions[edge.source];
          const targetPos = positions[edge.target];
          if (!sourcePos || !targetPos) return null;

          const isHighlighted =
            selectedNode === edge.source || selectedNode === edge.target;

          return (
            <line
              key={`edge-${i}`}
              x1={sourcePos.x}
              y1={sourcePos.y}
              x2={targetPos.x}
              y2={targetPos.y}
              stroke={isHighlighted ? "#2563eb" : "#d1d5db"}
              strokeWidth={isHighlighted ? 2 : 1}
              strokeOpacity={isHighlighted ? 1 : 0.5}
            />
          );
        })}

        {/* Nodes */}
        {graph.nodes.map((node, i) => {
          const pos = positions[i];
          if (!pos) return null;

          const isSelected = selectedNode === node.id;
          const baseRadius = 8 + node.degree * 0.8;
          const radius = isSelected ? baseRadius * 1.4 : baseRadius;

          return (
            <g
              key={`node-${node.id}`}
              className="cursor-pointer"
              onClick={(e) => {
                e.stopPropagation();
                if (!hasPanned) {
                  onNodeSelect(isSelected ? null : node.id);
                }
              }}
            >
              {/* Node circle */}
              <circle
                cx={pos.x}
                cy={pos.y}
                r={radius}
                fill={getNodeColor(node.importance)}
                stroke={isSelected ? "#2563eb" : "#fff"}
                strokeWidth={isSelected ? 2.5 : 1.5}
                style={{ transition: "r 0.2s ease-out, stroke 0.2s ease-out, stroke-width 0.2s ease-out" }}
              />

              {/* Node label */}
              <text
                x={pos.x}
                y={pos.y}
                textAnchor="middle"
                dominantBaseline="central"
                fontSize={isSelected ? 12 : 10}
                fontWeight={500}
                fill="#fff"
                className="pointer-events-none"
                style={{ transition: "font-size 0.2s ease-out" }}
              >
                {node.id}
              </text>
            </g>
          );
        })}
      </svg>

      {/* Legend */}
      <div className="absolute bottom-2 left-2 flex items-center gap-2 rounded bg-card/90 px-2 py-1 text-xs">
        <span className="text-muted-foreground">Importance:</span>
        <div className="flex items-center gap-1">
          <div className="h-3 w-3 rounded-full bg-[rgb(55,180,255)]" />
          <span>Faible</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="h-3 w-3 rounded-full bg-[rgb(255,80,55)]" />
          <span>Elevee</span>
        </div>
      </div>
    </div>
  );
}

// Node Details Panel
function NodeDetails({ node }: { node: Node | null }) {
  if (!node) {
    return (
      <div className="rounded-lg border border-dashed border-border bg-secondary/30 p-4 text-center text-sm text-muted-foreground">
        Cliquez sur un noeud pour voir ses details
      </div>
    );
  }

  return (
    <div className="space-y-3 rounded-lg border border-border bg-card p-4">
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium">Noeud {node.id}</span>
        <Badge
          variant="secondary"
          style={{
            backgroundColor: `rgba(${Math.round(node.importance * 200 + 55)}, 100, 100, 0.2)`,
          }}
        >
          Importance: {(node.importance * 100).toFixed(0)}%
        </Badge>
      </div>
      <Separator />
      <div className="grid grid-cols-2 gap-2 text-sm">
        <div>
          <span className="text-muted-foreground">Degre</span>
          <div className="font-mono font-medium">{node.degree}</div>
        </div>
        <div>
          <span className="text-muted-foreground">Clustering</span>
          <div className="font-mono font-medium">{node.clustering.toFixed(3)}</div>
        </div>
      </div>
    </div>
  );
}

// Main Page Component
export default function PlaygroundPage() {
  const [nNodes, setNNodes] = useState(15);
  const [edgeProb, setEdgeProb] = useState(0.4);
  const [graphType, setGraphType] = useState("erdos_renyi");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [selectedNode, setSelectedNode] = useState<number | null>(null);

  const runPrediction = useCallback(async () => {
    setLoading(true);
    setError(null);
    setSelectedNode(null);

    try {
      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          n_nodes: nNodes,
          edge_prob: edgeProb,
          graph_type: graphType,
        }),
      });

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || "Erreur serveur");
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Erreur de connexion au serveur");
    } finally {
      setLoading(false);
    }
  }, [nNodes, edgeProb, graphType]);

  const selectedNodeData = result?.graph.nodes.find((n) => n.id === selectedNode) || null;

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border bg-card">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
          <div className="flex items-center gap-4">
            <Link
              href="/"
              className="flex items-center gap-2 text-sm text-muted-foreground transition-colors hover:text-foreground"
            >
              <FontAwesomeIcon icon={faArrowLeft} className="h-3 w-3" />
              Retour
            </Link>
            <Separator orientation="vertical" className="h-6" />
            <div className="flex items-center gap-2">
              <FontAwesomeIcon icon={faAtom} className="h-5 w-5 text-primary" />
              <span className="text-lg font-semibold">Playground</span>
            </div>
          </div>
          <a
            href="https://github.com/PoCInnovation/FastQuantum"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 text-sm text-muted-foreground transition-colors hover:text-foreground"
          >
            <FontAwesomeIcon icon={faGithub} className="h-4 w-4" />
          </a>
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-6 py-8">
        <div className="mb-8">
          <h1 className="mb-2 text-2xl font-semibold">Testez le modele</h1>
          <p className="text-muted-foreground">
            Generez un graphe, executez l&apos;IA et visualisez les parametres QAOA predits.
          </p>
        </div>

        {/* Error banner */}
        {error && (
          <div className="mb-6 flex items-center gap-3 rounded-lg border border-destructive/50 bg-destructive/10 px-4 py-3 text-sm text-destructive">
            <FontAwesomeIcon icon={faTriangleExclamation} className="h-4 w-4" />
            {error}
          </div>
        )}

        <div className="grid gap-8 lg:grid-cols-[320px_1fr]">
          {/* Controls Panel */}
          <div className="space-y-6">
            <Card>
              <CardHeader className="pb-4">
                <CardTitle className="text-base">Configuration</CardTitle>
                <CardDescription>Parametres du graphe a generer</CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                {/* Graph Type */}
                <div className="space-y-2">
                  <Label>Type de graphe</Label>
                  <Select value={graphType} onValueChange={setGraphType}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {GRAPH_TYPES.map((type) => (
                        <SelectItem key={type.value} value={type.value}>
                          <div>
                            <div>{type.label}</div>
                            <div className="text-xs text-muted-foreground">
                              {type.description}
                            </div>
                          </div>
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                {/* Number of nodes */}
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <Label>Nombre de noeuds</Label>
                    <span className="font-mono text-sm">{nNodes}</span>
                  </div>
                  <Slider
                    value={[nNodes]}
                    onValueChange={([v]) => setNNodes(v)}
                    min={5}
                    max={40}
                    step={1}
                  />
                </div>

                {/* Edge probability */}
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <Label>Densite des aretes</Label>
                    <span className="font-mono text-sm">{edgeProb.toFixed(2)}</span>
                  </div>
                  <Slider
                    value={[edgeProb]}
                    onValueChange={([v]) => setEdgeProb(v)}
                    min={0.1}
                    max={0.8}
                    step={0.05}
                  />
                </div>

                <Separator />

                {/* Run button */}
                <Button
                  onClick={runPrediction}
                  disabled={loading}
                  className="w-full"
                  size="lg"
                >
                  {loading ? (
                    <>
                      <FontAwesomeIcon icon={faSpinner} className="mr-2 h-4 w-4 animate-spin" />
                      Execution...
                    </>
                  ) : (
                    <>
                      <FontAwesomeIcon icon={faPlay} className="mr-2 h-4 w-4" />
                      Executer l&apos;IA
                    </>
                  )}
                </Button>
              </CardContent>
            </Card>

            {/* Node details */}
            <div>
              <h3 className="mb-3 text-sm font-medium">Details du noeud</h3>
              <NodeDetails node={selectedNodeData} />
            </div>
          </div>

          {/* Results Panel */}
          <div className="space-y-6">
            {/* Graph visualization */}
            <Card>
              <CardHeader className="pb-4">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-base">Visualisation du graphe</CardTitle>
                  {result && (
                    <div className="flex gap-2 text-xs text-muted-foreground">
                      <Badge variant="outline">{result.graph.n_nodes} noeuds</Badge>
                      <Badge variant="outline">{result.graph.n_edges} aretes</Badge>
                    </div>
                  )}
                </div>
              </CardHeader>
              <CardContent>
                <GraphVisualization
                  graph={result?.graph || null}
                  selectedNode={selectedNode}
                  onNodeSelect={setSelectedNode}
                />
              </CardContent>
            </Card>

            {/* Predictions */}
            {result && (
              <div className="grid gap-4 sm:grid-cols-2">
                <Card>
                  <CardHeader className="pb-3">
                    <CardTitle className="flex items-center gap-2 text-base">
                      <span className="font-mono">gamma</span>
                      <span className="text-muted-foreground">(γ)</span>
                    </CardTitle>
                    <CardDescription>Parametre de phase QAOA</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-2">
                      {result.gamma.map((g, i) => (
                        <div key={i} className="flex items-center justify-between">
                          <span className="text-sm text-muted-foreground">
                            γ{i + 1}
                          </span>
                          <span className="font-mono text-lg font-medium">
                            {g.toFixed(4)}
                          </span>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader className="pb-3">
                    <CardTitle className="flex items-center gap-2 text-base">
                      <span className="font-mono">beta</span>
                      <span className="text-muted-foreground">(β)</span>
                    </CardTitle>
                    <CardDescription>Parametre de mixage QAOA</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-2">
                      {result.beta.map((b, i) => (
                        <div key={i} className="flex items-center justify-between">
                          <span className="text-sm text-muted-foreground">
                            β{i + 1}
                          </span>
                          <span className="font-mono text-lg font-medium">
                            {b.toFixed(4)}
                          </span>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </div>
            )}

            {/* Graph stats */}
            {result && (
              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="text-base">Statistiques du graphe</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-3 gap-4 text-center">
                    <div>
                      <div className="text-xs text-muted-foreground">Densite</div>
                      <div className="font-mono text-lg">
                        {result.graph.density.toFixed(3)}
                      </div>
                    </div>
                    <div>
                      <div className="text-xs text-muted-foreground">Clustering moy.</div>
                      <div className="font-mono text-lg">
                        {result.graph.avg_clustering.toFixed(3)}
                      </div>
                    </div>
                    <div>
                      <div className="text-xs text-muted-foreground">Degre moy.</div>
                      <div className="font-mono text-lg">
                        {result.graph.avg_degree.toFixed(1)}
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
