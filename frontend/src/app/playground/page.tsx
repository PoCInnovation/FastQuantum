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
  faSliders,
  faChartLine,
  faBolt,
} from "@fortawesome/free-solid-svg-icons";
import { faGithub } from "@fortawesome/free-brands-svg-icons";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
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
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs";
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
  probs: number[];
  predictions: number[];
  graph: GraphData;
  classical_predictions?: number[];
  qiskit_predictions?: number[];
  ai_execution_time: number;
  classical_execution_time?: number;
  qiskit_execution_time?: number;
  speedup?: number;
  qiskit_speedup?: number;
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

// Force-directed layout calculation
function calculateForceLayout(
  nodes: Node[],
  edges: Edge[],
  width: number,
  height: number
): { x: number; y: number }[] {
  if (nodes.length === 0) return [];

  const seededRandom = (seed: number) => {
    const x = Math.sin(seed) * 10000;
    return x - Math.floor(x);
  };

  const pos = nodes.map((_, i) => ({
    x: width / 2 + (seededRandom(i * 13) - 0.5) * width * 0.8,
    y: height / 2 + (seededRandom(i * 17 + 100) - 0.5) * height * 0.8,
  }));

  const iterations = 100;
  const k = Math.sqrt((width * height) / nodes.length) * 0.8;

  for (let iter = 0; iter < iterations; iter++) {
    const temp = 1 - iter / iterations;

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

    const margin = 40;
    for (let i = 0; i < nodes.length; i++) {
      pos[i].x = Math.max(margin, Math.min(width - margin, pos[i].x));
      pos[i].y = Math.max(margin, Math.min(height - margin, pos[i].y));
    }
  }

  return pos;
}

function GraphVisualization({
  graph,
  selectedNode,
  result,
  onNodeSelect,
}: {
  graph: GraphData | null;
  selectedNode: number | null;
  result: PredictionResult | null;
  onNodeSelect: (id: number | null) => void;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const [dimensions, setDimensions] = useState({ width: 500, height: 450 });
  const [viewBox, setViewBox] = useState({ x: 0, y: 0, width: 500, height: 450 });
  const [isPanning, setIsPanning] = useState(false);
  const [hasPanned, setHasPanned] = useState(false);
  const [startPan, setStartPan] = useState({ x: 0, y: 0 });
  const [zoom, setZoom] = useState(1);

  useEffect(() => {
    if (containerRef.current) {
      const rect = containerRef.current.getBoundingClientRect();
      setDimensions({ width: rect.width, height: 450 });
      setViewBox({ x: 0, y: 0, width: rect.width, height: 450 });
    }
  }, []);

  useEffect(() => {
    if (graph) {
      setViewBox({ x: 0, y: 0, width: dimensions.width, height: dimensions.height });
      setZoom(1);
    }
  }, [graph, dimensions.width, dimensions.height]);

  const handleWheel = useCallback(
    (e: React.WheelEvent) => {
      e.preventDefault();
      const scaleAmount = e.deltaY > 0 ? 1.1 : 0.9;
      const newZoom = Math.min(Math.max(zoom * scaleAmount, 0.2), 5);
      const svg = svgRef.current;
      if (!svg) return;
      const rect = svg.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;
      const viewBoxMouseX = viewBox.x + (mouseX / dimensions.width) * viewBox.width;
      const viewBoxMouseY = viewBox.y + (mouseY / dimensions.height) * viewBox.height;
      const newWidth = dimensions.width * (1 / newZoom);
      const newHeight = dimensions.height * (1 / newZoom);
      const newX = viewBoxMouseX - (mouseX / dimensions.width) * newWidth;
      const newY = viewBoxMouseY - (mouseY / dimensions.height) * newHeight;
      setViewBox({ x: newX, y: newY, width: newWidth, height: newHeight });
      setZoom(newZoom);
    },
    [zoom, viewBox, dimensions]
  );

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (e.button === 0) {
      setIsPanning(true);
      setHasPanned(false);
      setStartPan({ x: e.clientX, y: e.clientY });
    }
  }, []);

  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      if (!isPanning) return;
      const dx = (e.clientX - startPan.x) * (viewBox.width / dimensions.width);
      const dy = (e.clientY - startPan.y) * (viewBox.height / dimensions.height);
      if (Math.abs(e.clientX - startPan.x) > 3 || Math.abs(e.clientY - startPan.y) > 3) {
        setHasPanned(true);
      }
      setViewBox((prev) => ({ ...prev, x: prev.x - dx, y: prev.y - dy }));
      setStartPan({ x: e.clientX, y: e.clientY });
    },
    [isPanning, startPan, viewBox, dimensions]
  );

  const handleMouseUp = useCallback(() => {
    setIsPanning(false);
    setTimeout(() => setHasPanned(false), 100);
  }, []);

  const handleMouseLeave = useCallback(() => {
    setIsPanning(false);
    setHasPanned(false);
  }, []);

  const resetView = useCallback(() => {
    setViewBox({ x: 0, y: 0, width: dimensions.width, height: dimensions.height });
    setZoom(1);
  }, [dimensions]);

  const positions = useMemo(() => {
    if (!graph) return [];
    return calculateForceLayout(graph.nodes, graph.edges, dimensions.width, dimensions.height);
  }, [graph, dimensions.width, dimensions.height]);

  if (!graph) {
    return (
      <div
        ref={containerRef}
        className="flex h-[450px] items-center justify-center rounded-xl border border-dashed border-border/50 bg-secondary/20"
      >
        <div className="text-center text-muted-foreground">
          <FontAwesomeIcon
            icon={faCircleNodes}
            className="mb-3 h-10 w-10 opacity-20"
          />
          <p className="text-sm">Generez un graphe pour commencer</p>
        </div>
      </div>
    );
  }

  const getNodeColor = (id: number) => {
    if (!result) {
      const node = graph.nodes.find((n) => n.id === id);
      const importance = node ? node.importance : 0.5;
      // Cyan to violet gradient based on importance
      const r = Math.round(importance * 139 + (1 - importance) * 0);
      const g = Math.round(importance * 92 + (1 - importance) * 212);
      const b = Math.round(importance * 246 + (1 - importance) * 255);
      return `rgb(${r}, ${g}, ${b})`;
    }
    const pred = result.predictions[id];
    return pred === 1 ? "#f97316" : "#00d4ff";
  };

  const getNodeGlow = (id: number) => {
    if (!result) {
      const node = graph.nodes.find((n) => n.id === id);
      const importance = node ? node.importance : 0.5;
      return importance > 0.6 ? "rgba(139, 92, 246, 0.5)" : "rgba(0, 212, 255, 0.3)";
    }
    const pred = result.predictions[id];
    return pred === 1 ? "rgba(249, 115, 22, 0.4)" : "rgba(0, 212, 255, 0.4)";
  };

  return (
    <div ref={containerRef} className="relative">
      {/* Zoom controls */}
      <div className="absolute right-3 top-3 z-10 flex flex-col gap-1">
        {[
          { icon: faMagnifyingGlassPlus, title: "Zoom avant", action: () => {
            const newZoom = Math.min(zoom * 1.3, 5);
            const centerX = viewBox.x + viewBox.width / 2;
            const centerY = viewBox.y + viewBox.height / 2;
            const newWidth = dimensions.width * (1 / newZoom);
            const newHeight = dimensions.height * (1 / newZoom);
            setViewBox({ x: centerX - newWidth / 2, y: centerY - newHeight / 2, width: newWidth, height: newHeight });
            setZoom(newZoom);
          }},
          { icon: faMagnifyingGlassMinus, title: "Zoom arriere", action: () => {
            const newZoom = Math.max(zoom * 0.7, 0.2);
            const centerX = viewBox.x + viewBox.width / 2;
            const centerY = viewBox.y + viewBox.height / 2;
            const newWidth = dimensions.width * (1 / newZoom);
            const newHeight = dimensions.height * (1 / newZoom);
            setViewBox({ x: centerX - newWidth / 2, y: centerY - newHeight / 2, width: newWidth, height: newHeight });
            setZoom(newZoom);
          }},
          { icon: faExpand, title: "Reset", action: resetView },
        ].map((btn, i) => (
          <button
            key={i}
            onClick={btn.action}
            className="flex h-8 w-8 items-center justify-center rounded-lg glass text-muted-foreground hover:text-primary transition-colors"
            title={btn.title}
          >
            <FontAwesomeIcon icon={btn.icon} className="h-3.5 w-3.5" />
          </button>
        ))}
      </div>

      {/* Zoom indicator */}
      <div className="absolute left-3 top-3 z-10 glass rounded-lg px-2 py-1 text-xs font-mono text-primary/70">
        {Math.round(zoom * 100)}%
      </div>

      <svg
        ref={svgRef}
        width={dimensions.width}
        height={dimensions.height}
        viewBox={`${viewBox.x} ${viewBox.y} ${viewBox.width} ${viewBox.height}`}
        className={`rounded-xl border border-border/50 bg-[#080814] ${
          isPanning ? "cursor-grabbing" : "cursor-grab"
        }`}
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseLeave}
      >
        {/* Background grid pattern */}
        <defs>
          <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
            <path d="M 40 0 L 0 0 0 40" fill="none" stroke="rgba(30,41,59,0.3)" strokeWidth="0.5" />
          </pattern>
          <filter id="glow">
            <feGaussianBlur stdDeviation="3" result="coloredBlur" />
            <feMerge>
              <feMergeNode in="coloredBlur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>
        <rect width="100%" height="100%" fill="url(#grid)" />

        {/* Edges */}
        {graph.edges.map((edge, i) => {
          const sourcePos = positions[edge.source];
          const targetPos = positions[edge.target];
          if (!sourcePos || !targetPos) return null;
          const isHighlighted = selectedNode === edge.source || selectedNode === edge.target;

          return (
            <line
              key={`edge-${i}`}
              x1={sourcePos.x}
              y1={sourcePos.y}
              x2={targetPos.x}
              y2={targetPos.y}
              stroke={isHighlighted ? "#00d4ff" : "rgba(100, 116, 139, 0.25)"}
              strokeWidth={isHighlighted ? 2 : 1}
              strokeOpacity={isHighlighted ? 0.8 : 0.5}
              style={{ transition: "stroke 0.3s, stroke-width 0.3s" }}
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
          const color = getNodeColor(node.id);
          const glow = getNodeGlow(node.id);

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
              filter={isSelected ? "url(#glow)" : undefined}
            >
              {/* Glow circle */}
              <circle
                cx={pos.x}
                cy={pos.y}
                r={radius + 4}
                fill="none"
                stroke={glow}
                strokeWidth={isSelected ? 2 : 0}
                opacity={isSelected ? 0.6 : 0}
                style={{ transition: "all 0.3s ease" }}
              />
              {/* Node circle */}
              <circle
                cx={pos.x}
                cy={pos.y}
                r={radius}
                fill={color}
                stroke={isSelected ? "#ffffff" : "rgba(255,255,255,0.15)"}
                strokeWidth={isSelected ? 2 : 1}
                style={{
                  transition: "r 0.2s ease-out, stroke 0.2s ease-out, stroke-width 0.2s ease-out, fill 0.3s ease",
                }}
              />
              {/* Node label */}
              <text
                x={pos.x}
                y={pos.y}
                textAnchor="middle"
                dominantBaseline="central"
                fontSize={isSelected ? 11 : 9}
                fontWeight={600}
                fill={isSelected ? "#ffffff" : "rgba(255,255,255,0.9)"}
                className="pointer-events-none"
                fontFamily="var(--font-mono)"
                style={{ transition: "font-size 0.2s ease-out" }}
              >
                {node.id}
              </text>
            </g>
          );
        })}
      </svg>

      {/* Legend */}
      <div className="absolute bottom-3 left-3 flex items-center gap-3 glass rounded-lg px-3 py-2 text-xs">
        {result ? (
          <>
            <span className="font-medium text-muted-foreground">Partition:</span>
            <div className="flex items-center gap-1.5">
              <div className="h-2.5 w-2.5 rounded-full bg-[#00d4ff] shadow-[0_0_6px_rgba(0,212,255,0.5)]" />
              <span className="text-muted-foreground">Classe 0</span>
            </div>
            <div className="flex items-center gap-1.5">
              <div className="h-2.5 w-2.5 rounded-full bg-orange-500 shadow-[0_0_6px_rgba(249,115,22,0.5)]" />
              <span className="text-muted-foreground">Classe 1</span>
            </div>
          </>
        ) : (
          <>
            <span className="text-muted-foreground">Importance:</span>
            <div className="flex items-center gap-1">
              <div className="h-2.5 w-2.5 rounded-full bg-[#00d4ff]" />
              <span className="text-muted-foreground">Faible</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="h-2.5 w-2.5 rounded-full bg-[#8b5cf6]" />
              <span className="text-muted-foreground">Elevee</span>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

// Node Details Panel
function NodeDetails({
  node,
  result,
}: {
  node: Node | null;
  result: PredictionResult | null;
}) {
  if (!node) {
    return (
      <div className="rounded-xl border border-dashed border-border/50 bg-secondary/10 p-6 text-center text-sm text-muted-foreground">
        <FontAwesomeIcon
          icon={faCircleNodes}
          className="mb-2 h-6 w-6 opacity-20"
        />
        <p>Cliquez sur un noeud</p>
      </div>
    );
  }

  const prob = result?.probs?.[node.id];
  const predClass = result?.predictions?.[node.id];

  return (
    <div className="space-y-3 rounded-xl border border-border/50 bg-card/80 p-4">
      <div className="flex items-center justify-between">
        <span className="text-sm font-semibold">
          Noeud{" "}
          <span className="font-mono text-primary">{node.id}</span>
        </span>
        {result ? (
          <Badge
            className={
              predClass === 1
                ? "bg-orange-500/20 text-orange-400 border border-orange-500/30 hover:bg-orange-500/30"
                : "bg-[#00d4ff]/20 text-[#00d4ff] border border-[#00d4ff]/30 hover:bg-[#00d4ff]/30"
            }
          >
            Classe {predClass}
          </Badge>
        ) : (
          <Badge
            variant="outline"
            className="border-accent/30 text-accent"
          >
            {(node.importance * 100).toFixed(0)}%
          </Badge>
        )}
      </div>
      <Separator className="bg-border/30" />
      <div className="grid grid-cols-2 gap-3 text-sm">
        <div>
          <span className="text-[11px] uppercase tracking-wider text-muted-foreground/60">
            Degre
          </span>
          <div className="font-mono font-semibold text-foreground">
            {node.degree}
          </div>
        </div>
        <div>
          <span className="text-[11px] uppercase tracking-wider text-muted-foreground/60">
            Clustering
          </span>
          <div className="font-mono font-semibold text-foreground">
            {node.clustering.toFixed(3)}
          </div>
        </div>
        {result && prob !== undefined && (
          <div className="col-span-2 mt-1">
            <span className="text-[11px] uppercase tracking-wider text-muted-foreground/60">
              Confiance IA
            </span>
            <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-secondary">
              <div
                className="h-full rounded-full bg-primary transition-all duration-500"
                style={{ width: `${prob * 100}%` }}
              />
            </div>
            <div className="mt-1 text-right text-xs font-mono text-primary">
              {(prob * 100).toFixed(1)}%
            </div>
          </div>
        )}
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
      setError(
        err instanceof Error ? err.message : "Erreur de connexion au serveur"
      );
    } finally {
      setLoading(false);
    }
  }, [nNodes, edgeProb, graphType]);

  const selectedNodeData =
    result?.graph.nodes.find((n) => n.id === selectedNode) || null;

  return (
    <div className="min-h-screen bg-background relative">
      {/* Background effects */}
      <div className="fixed inset-0 bg-grid opacity-20 pointer-events-none" />

      {/* Header */}
      <header className="sticky top-0 z-50 glass">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-3">
          <div className="flex items-center gap-4">
            <Link
              href="/"
              className="group flex items-center gap-2 text-sm text-muted-foreground transition-colors hover:text-primary"
            >
              <FontAwesomeIcon
                icon={faArrowLeft}
                className="h-3 w-3 transition-transform group-hover:-translate-x-0.5"
              />
              Retour
            </Link>
            <Separator orientation="vertical" className="h-5 bg-border/50" />
            <div className="flex items-center gap-2">
              <div className="relative flex h-7 w-7 items-center justify-center">
                <div className="absolute inset-0 rounded-md bg-primary/10" />
                <FontAwesomeIcon
                  icon={faAtom}
                  className="relative h-3.5 w-3.5 text-primary"
                />
              </div>
              <span className="font-semibold tracking-tight">
                Play<span className="text-primary">ground</span>
              </span>
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

      <main className="relative mx-auto max-w-7xl px-6 py-8">
        <div className="mb-8">
          <h1 className="mb-2 text-2xl font-bold tracking-tight">
            Testez le <span className="gradient-text">modele</span>
          </h1>
          <p className="text-sm text-muted-foreground max-w-lg">
            Generez un graphe, executez l&apos;IA et visualisez les parametres
            QAOA predits en temps reel.
          </p>
        </div>

        {/* Error banner */}
        {error && (
          <div className="mb-6 flex items-center gap-3 rounded-xl border border-destructive/30 bg-destructive/5 px-4 py-3 text-sm text-destructive animate-fade-in-up">
            <FontAwesomeIcon
              icon={faTriangleExclamation}
              className="h-4 w-4"
            />
            {error}
          </div>
        )}

        <Tabs defaultValue="parameters" className="w-full">
          <TabsList className="mb-6 grid w-full grid-cols-2 bg-secondary/50 border border-border/50 p-1 rounded-xl">
            <TabsTrigger
              value="parameters"
              className="flex items-center gap-2 rounded-lg data-[state=active]:bg-primary/10 data-[state=active]:text-primary data-[state=active]:shadow-none transition-all"
            >
              <FontAwesomeIcon icon={faSliders} className="h-3.5 w-3.5" />
              Solution MaxCut
            </TabsTrigger>
            <TabsTrigger
              value="results"
              className="flex items-center gap-2 rounded-lg data-[state=active]:bg-primary/10 data-[state=active]:text-primary data-[state=active]:shadow-none transition-all"
            >
              <FontAwesomeIcon icon={faChartLine} className="h-3.5 w-3.5" />
              Visualisation
            </TabsTrigger>
          </TabsList>

          {/* TAB 1: Parameter Prediction */}
          <TabsContent value="parameters">
            <div className="grid gap-8 lg:grid-cols-2">
              {/* Configuration Panel */}
              <Card className="card-glow border-border/50 bg-card/80">
                <CardHeader className="pb-4">
                  <CardTitle className="text-base font-semibold flex items-center gap-2">
                    <FontAwesomeIcon
                      icon={faSliders}
                      className="h-3.5 w-3.5 text-primary"
                    />
                    Configuration du graphe
                  </CardTitle>
                  <CardDescription>
                    Definissez les parametres du graphe a analyser
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                  {/* Graph Type */}
                  <div className="space-y-2">
                    <Label className="text-xs font-semibold uppercase tracking-wider text-muted-foreground/70">
                      Type de graphe
                    </Label>
                    <Select value={graphType} onValueChange={setGraphType}>
                      <SelectTrigger className="bg-secondary/50 border-border/50 hover:border-primary/30 transition-colors">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent className="bg-card border-border/50">
                        {GRAPH_TYPES.map((type) => (
                          <SelectItem key={type.value} value={type.value}>
                            <div>
                              <div className="font-medium">{type.label}</div>
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
                      <Label className="text-xs font-semibold uppercase tracking-wider text-muted-foreground/70">
                        Nombre de noeuds
                      </Label>
                      <span className="font-mono text-sm font-bold text-primary">
                        {nNodes}
                      </span>
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
                      <Label className="text-xs font-semibold uppercase tracking-wider text-muted-foreground/70">
                        Densite des aretes
                      </Label>
                      <span className="font-mono text-sm font-bold text-primary">
                        {edgeProb.toFixed(2)}
                      </span>
                    </div>
                    <Slider
                      value={[edgeProb]}
                      onValueChange={([v]) => setEdgeProb(v)}
                      min={0.1}
                      max={0.8}
                      step={0.05}
                    />
                  </div>

                  <Separator className="bg-border/30" />

                  {/* Run button */}
                  <Button
                    onClick={runPrediction}
                    disabled={loading}
                    className="w-full bg-primary text-primary-foreground hover:bg-primary/90 glow-cyan font-semibold tracking-wide"
                    size="lg"
                  >
                    {loading ? (
                      <>
                        <FontAwesomeIcon
                          icon={faSpinner}
                          className="mr-2 h-4 w-4 animate-spin"
                        />
                        Execution...
                      </>
                    ) : (
                      <>
                        <FontAwesomeIcon
                          icon={faPlay}
                          className="mr-2 h-3.5 w-3.5"
                        />
                        Resoudre
                      </>
                    )}
                  </Button>
                </CardContent>
              </Card>

              <div className="space-y-4">
                {result ? (
                  <>
                    {/* Performance Comparison */}
                    {result.speedup && result.classical_execution_time && (
                      <Card className="border-primary/20 bg-primary/5 card-glow">
                        <CardHeader className="pb-2">
                          <CardTitle className="flex items-center gap-2 text-base">
                            <FontAwesomeIcon
                              icon={faBolt}
                              className="h-4 w-4 text-primary"
                            />
                            <span className="text-primary font-semibold">
                              Performance
                            </span>
                          </CardTitle>
                        </CardHeader>
                        <CardContent>
                          <div
                            className={`grid gap-4 ${
                              result.qiskit_execution_time
                                ? "grid-cols-2"
                                : "grid-cols-1"
                            }`}
                          >
                            <div>
                              <div className="text-[10px] font-semibold uppercase tracking-widest text-muted-foreground/60">
                                Temps IA (GNN)
                              </div>
                              <div className="mt-1 font-mono text-2xl font-bold text-primary">
                                {(result.ai_execution_time * 1000).toFixed(2)}{" "}
                                <span className="text-sm font-normal text-muted-foreground">
                                  ms
                                </span>
                              </div>
                            </div>
                            {result.qiskit_execution_time && (
                              <div>
                                <div className="text-[10px] font-semibold uppercase tracking-widest text-muted-foreground/60">
                                  Temps Qiskit
                                </div>
                                <div className="mt-1 font-mono text-2xl font-bold text-accent">
                                  {(result.qiskit_execution_time * 1000).toFixed(2)}{" "}
                                  <span className="text-sm font-normal text-muted-foreground">
                                    ms
                                  </span>
                                </div>
                              </div>
                            )}
                          </div>

                          {result.qiskit_speedup && (
                            <>
                              <Separator className="my-4 bg-primary/10" />
                              <div className="flex items-center justify-between">
                                <span className="text-sm font-medium text-muted-foreground">
                                  Acceleration vs Qiskit
                                </span>
                                <Badge className="bg-primary/20 text-primary border border-primary/30 px-3 py-1 text-base font-mono font-bold hover:bg-primary/30">
                                  x{result.qiskit_speedup.toFixed(1)}
                                </Badge>
                              </div>
                            </>
                          )}
                        </CardContent>
                      </Card>
                    )}

                    <div
                      className={`grid gap-4 ${
                        result.qiskit_predictions
                          ? "md:grid-cols-2"
                          : "md:grid-cols-1"
                      }`}
                    >
                      {/* AI Results */}
                      <div className="space-y-3">
                        <h3 className="flex items-center gap-2 text-sm font-semibold text-primary">
                          <FontAwesomeIcon
                            icon={faAtom}
                            className="h-3.5 w-3.5"
                          />
                          Predictions IA
                        </h3>
                        <Card className="border-primary/20 bg-primary/5">
                          <CardHeader className="pb-2">
                            <CardTitle className="text-xs uppercase tracking-wider text-muted-foreground/70">
                              Bitstring
                            </CardTitle>
                          </CardHeader>
                          <CardContent className="pb-3">
                            <div className="flex flex-wrap gap-1">
                              {result.predictions.map((p, i) => (
                                <span
                                  key={i}
                                  className={`inline-flex h-6 w-6 items-center justify-center rounded font-mono text-xs font-bold ${
                                    p === 1
                                      ? "bg-orange-500/20 text-orange-400 border border-orange-500/30"
                                      : "bg-[#00d4ff]/15 text-[#00d4ff] border border-[#00d4ff]/30"
                                  }`}
                                >
                                  {p}
                                </span>
                              ))}
                            </div>
                          </CardContent>
                        </Card>
                      </div>

                      {result.qiskit_predictions && (
                        <div className="space-y-3">
                          <h3 className="flex items-center gap-2 text-sm font-semibold text-accent">
                            <FontAwesomeIcon
                              icon={faAtom}
                              className="h-3.5 w-3.5"
                            />
                            Qiskit QAOA
                          </h3>
                          <Card className="border-accent/20 bg-accent/5">
                            <CardHeader className="pb-2">
                              <CardTitle className="text-xs uppercase tracking-wider text-muted-foreground/70">
                                Simulation
                              </CardTitle>
                            </CardHeader>
                            <CardContent className="pb-3">
                              <div className="flex flex-wrap gap-1">
                                {result.qiskit_predictions.map((p, i) => (
                                  <span
                                    key={i}
                                    className={`inline-flex h-6 w-6 items-center justify-center rounded font-mono text-xs font-bold border ${
                                      p === 1
                                        ? "bg-accent/20 text-accent border-accent/30"
                                        : "bg-accent/5 text-accent/60 border-accent/15"
                                    }`}
                                  >
                                    {p}
                                  </span>
                                ))}
                              </div>
                            </CardContent>
                          </Card>
                        </div>
                      )}
                    </div>
                  </>
                ) : (
                  <Card className="border-dashed border-border/50 bg-card/40">
                    <CardContent className="flex h-[400px] items-center justify-center">
                      <div className="text-center text-muted-foreground">
                        <div className="relative mx-auto mb-4 h-16 w-16">
                          <div className="absolute inset-0 rounded-2xl bg-primary/5 border border-primary/10" />
                          <FontAwesomeIcon
                            icon={faSliders}
                            className="absolute inset-0 m-auto h-7 w-7 opacity-20"
                          />
                        </div>
                        <p className="text-sm font-medium">
                          Configurez et executez
                        </p>
                        <p className="mt-1 text-xs text-muted-foreground/60">
                          Comparaison IA vs Qiskit
                        </p>
                      </div>
                    </CardContent>
                  </Card>
                )}
              </div>
            </div>
          </TabsContent>

          {/* TAB 2: Visualization */}
          <TabsContent value="results">
            <div className="grid gap-8 lg:grid-cols-[1fr_280px]">
              {/* Graph visualization */}
              <div className="space-y-6">
                <Card className="card-glow border-border/50 bg-card/80 overflow-hidden">
                  <CardHeader className="pb-4">
                    <div className="flex items-center justify-between">
                      <CardTitle className="text-base font-semibold flex items-center gap-2">
                        <FontAwesomeIcon
                          icon={faCircleNodes}
                          className="h-3.5 w-3.5 text-primary"
                        />
                        Visualisation
                      </CardTitle>
                      {result && (
                        <div className="flex gap-2 text-xs">
                          <Badge
                            variant="outline"
                            className="border-primary/20 text-primary font-mono"
                          >
                            {result.graph.n_nodes} noeuds
                          </Badge>
                          <Badge
                            variant="outline"
                            className="border-accent/20 text-accent font-mono"
                          >
                            {result.graph.n_edges} aretes
                          </Badge>
                        </div>
                      )}
                    </div>
                  </CardHeader>
                  <CardContent>
                    <GraphVisualization
                      graph={result?.graph || null}
                      selectedNode={selectedNode}
                      result={result}
                      onNodeSelect={setSelectedNode}
                    />
                  </CardContent>
                </Card>

                {/* Graph stats */}
                {result && (
                  <div className="grid grid-cols-3 gap-4">
                    {[
                      {
                        label: "Densite",
                        value: result.graph.density.toFixed(3),
                      },
                      {
                        label: "Clustering moy.",
                        value: result.graph.avg_clustering.toFixed(3),
                      },
                      {
                        label: "Degre moy.",
                        value: result.graph.avg_degree.toFixed(1),
                      },
                    ].map((stat) => (
                      <div
                        key={stat.label}
                        className="rounded-xl border border-border/50 bg-card/60 p-4 text-center card-glow"
                      >
                        <div className="text-[10px] font-semibold uppercase tracking-widest text-muted-foreground/60">
                          {stat.label}
                        </div>
                        <div className="mt-1 font-mono text-xl font-bold text-foreground">
                          {stat.value}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* Right sidebar */}
              <div className="space-y-6">
                <div>
                  <h3 className="mb-3 text-xs font-semibold uppercase tracking-wider text-muted-foreground/70">
                    Details du noeud
                  </h3>
                  <NodeDetails node={selectedNodeData} result={result} />
                </div>

                {result && (
                  <Card className="border-border/50 bg-card/80 card-glow">
                    <CardHeader className="pb-3">
                      <CardTitle className="text-sm font-semibold">
                        Solution MaxCut
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div>
                        <div className="mb-2 text-[10px] font-semibold uppercase tracking-widest text-muted-foreground/60">
                          Bitstring IA
                        </div>
                        <div className="rounded-lg bg-primary/5 border border-primary/20 px-3 py-2 text-center font-mono text-sm font-bold tracking-[0.2em] text-primary break-all">
                          {result.predictions.join("")}
                        </div>
                      </div>

                      {result.qiskit_predictions && (
                        <>
                          <Separator className="bg-border/30" />
                          <div>
                            <div className="mb-2 text-[10px] font-semibold uppercase tracking-widest text-muted-foreground/60">
                              Bitstring Qiskit
                            </div>
                            <div className="rounded-lg bg-accent/5 border border-accent/20 px-3 py-2 text-center font-mono text-sm font-bold tracking-[0.2em] text-accent break-all">
                              {result.qiskit_predictions.join("")}
                            </div>
                          </div>

                          <Separator className="bg-border/30" />
                          <div>
                            <div className="mb-2 text-[10px] font-semibold uppercase tracking-widest text-muted-foreground/60">
                              Similarite
                            </div>
                            <div className="flex items-center justify-between">
                              <span className="text-sm">
                                {result.predictions.reduce(
                                  (acc, p, i) =>
                                    acc +
                                    (p === result.qiskit_predictions![i]
                                      ? 1
                                      : 0),
                                  0
                                )}{" "}
                                / {result.predictions.length}
                              </span>
                              <Badge className="bg-primary/10 text-primary border border-primary/20 font-mono font-bold hover:bg-primary/20">
                                {Math.round(
                                  (result.predictions.reduce(
                                    (acc, p, i) =>
                                      acc +
                                      (p === result.qiskit_predictions![i]
                                        ? 1
                                        : 0),
                                    0
                                  ) /
                                    result.predictions.length) *
                                    100
                                )}
                                %
                              </Badge>
                            </div>
                          </div>
                        </>
                      )}
                    </CardContent>
                  </Card>
                )}

                {!result && (
                  <Card className="border-dashed border-border/50 bg-card/40">
                    <CardContent className="py-10 text-center">
                      <FontAwesomeIcon
                        icon={faChartLine}
                        className="mb-3 h-8 w-8 text-muted-foreground/15"
                      />
                      <p className="text-xs text-muted-foreground">
                        Executez une prediction
                      </p>
                    </CardContent>
                  </Card>
                )}
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </main>
    </div>
  );
}
