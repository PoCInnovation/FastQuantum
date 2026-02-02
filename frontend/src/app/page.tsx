"use client";

import { useState, useEffect } from "react";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import {
  faAtom,
  faBrain,
  faChartLine,
  faDatabase,
  faCircleInfo,
  faArrowRight,
  faPlay,
} from "@fortawesome/free-solid-svg-icons";
import { faGithub } from "@fortawesome/free-brands-svg-icons";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Separator } from "@/components/ui/separator";
import Link from "next/link";

// Types for metrics
interface ModelMetrics {
  epoch: number;
  val_loss: number;
  gamma_mae: number;
  beta_mae: number;
  p_layers: number;
  model_type: string;
  total_parameters: number;
}

interface DatasetStats {
  n_samples: number;
  nodes: { min: number; max: number; mean: number; std: number };
  edges: { min: number; max: number; mean: number; std: number };
  features: { means: number[]; stds: number[] };
  gamma: { min: number; max: number; mean: number; std: number };
  beta: { min: number; max: number; mean: number; std: number };
}

interface Feature {
  name: string;
  key: string;
  description: string;
  formula: string;
  interpretation: string;
}

interface Metrics {
  model: ModelMetrics | null;
  training: DatasetStats | null;
  validation: DatasetStats | null;
  features: Feature[];
}

export default function Home() {
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedFeature, setSelectedFeature] = useState<number>(0);

  useEffect(() => {
    fetch("/metrics.json")
      .then((res) => res.json())
      .then((data) => {
        setMetrics(data);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <div className="text-muted-foreground">Chargement...</div>
      </div>
    );
  }

  if (!metrics) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <div className="text-muted-foreground">
          Erreur: Lancez `python scripts/extract_metrics.py` pour generer les metriques.
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border bg-card">
        <div className="mx-auto flex max-w-5xl items-center justify-between px-6 py-4">
          <div className="flex items-center gap-3">
            <FontAwesomeIcon icon={faAtom} className="h-5 w-5 text-primary" />
            <span className="text-lg font-semibold">FastQuantum</span>
          </div>
          <div className="flex items-center gap-4">
            <Link
              href="/playground"
              className="flex items-center gap-2 text-sm font-medium text-primary transition-colors hover:text-primary/80"
            >
              <FontAwesomeIcon icon={faPlay} className="h-3 w-3" />
              Playground
            </Link>
            <a
              href="https://github.com/PoCInnovation/FastQuantum"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 text-sm text-muted-foreground transition-colors hover:text-foreground"
            >
              <FontAwesomeIcon icon={faGithub} className="h-4 w-4" />
              GitHub
            </a>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-5xl px-6 py-12">
        {/* Hero */}
        <section className="mb-16">
          <h1 className="mb-4 text-3xl font-semibold tracking-tight">
            Prediction des parametres QAOA
          </h1>
          <p className="mb-6 max-w-2xl text-lg text-muted-foreground">
            Utilisation de Graph Neural Networks (GAT) pour predire les parametres optimaux
            gamma et beta de l&apos;algorithme QAOA, accelerant l&apos;optimisation de circuits quantiques.
          </p>
          <Link href="/playground">
            <Button size="lg" className="gap-2">
              <FontAwesomeIcon icon={faPlay} className="h-4 w-4" />
              Tester le modele
            </Button>
          </Link>
        </section>

        {/* Model Performance */}
        {metrics.model && (
          <section className="mb-16">
            <div className="mb-6 flex items-center gap-2">
              <FontAwesomeIcon icon={faBrain} className="h-4 w-4 text-muted-foreground" />
              <h2 className="text-lg font-medium">Performance du modele</h2>
              <Badge variant="secondary" className="ml-2">
                {metrics.model.model_type}
              </Badge>
            </div>

            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
              <MetricCard
                label="Validation Loss"
                value={metrics.model.val_loss.toFixed(6)}
                description="Erreur quadratique moyenne sur le jeu de validation"
              />
              <MetricCard
                label="Gamma MAE"
                value={metrics.model.gamma_mae.toFixed(4)}
                description="Erreur absolue moyenne sur le parametre gamma"
              />
              <MetricCard
                label="Beta MAE"
                value={metrics.model.beta_mae.toFixed(4)}
                description="Erreur absolue moyenne sur le parametre beta"
              />
              <MetricCard
                label="Parametres"
                value={metrics.model.total_parameters.toLocaleString()}
                description={`Entraines en ${metrics.model.epoch} epochs`}
              />
            </div>
          </section>
        )}

        {/* Dataset Statistics */}
        {metrics.training && metrics.validation && (
          <section className="mb-16">
            <div className="mb-6 flex items-center gap-2">
              <FontAwesomeIcon icon={faDatabase} className="h-4 w-4 text-muted-foreground" />
              <h2 className="text-lg font-medium">Donnees d&apos;entrainement</h2>
            </div>

            <Tabs defaultValue="training" className="w-full">
              <TabsList className="mb-4">
                <TabsTrigger value="training">
                  Entrainement ({metrics.training.n_samples})
                </TabsTrigger>
                <TabsTrigger value="validation">
                  Validation ({metrics.validation.n_samples})
                </TabsTrigger>
              </TabsList>

              <TabsContent value="training">
                <DatasetCard data={metrics.training} />
              </TabsContent>

              <TabsContent value="validation">
                <DatasetCard data={metrics.validation} />
              </TabsContent>
            </Tabs>
          </section>
        )}

        {/* Node Features */}
        {metrics.features && metrics.training && (
          <section className="mb-16">
            <div className="mb-6 flex items-center gap-2">
              <FontAwesomeIcon icon={faChartLine} className="h-4 w-4 text-muted-foreground" />
              <h2 className="text-lg font-medium">Features des noeuds</h2>
              <span className="text-sm text-muted-foreground">
                (7 features par noeud)
              </span>
            </div>

            <div className="grid gap-6 lg:grid-cols-[280px_1fr]">
              {/* Feature list */}
              <div className="space-y-1">
                {metrics.features.map((feature, i) => (
                  <button
                    key={feature.key}
                    onClick={() => setSelectedFeature(i)}
                    className={`flex w-full items-center justify-between rounded-lg px-3 py-2 text-left text-sm transition-colors ${
                      selectedFeature === i
                        ? "bg-primary text-primary-foreground"
                        : "hover:bg-secondary"
                    }`}
                  >
                    <span>{feature.name}</span>
                    <FontAwesomeIcon
                      icon={faArrowRight}
                      className={`h-3 w-3 ${selectedFeature === i ? "opacity-100" : "opacity-0"}`}
                    />
                  </button>
                ))}
              </div>

              {/* Feature detail */}
              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="text-base">
                    {metrics.features[selectedFeature].name}
                  </CardTitle>
                  <CardDescription>
                    {metrics.features[selectedFeature].description}
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div>
                    <div className="mb-1 text-xs font-medium uppercase tracking-wider text-muted-foreground">
                      Formule
                    </div>
                    <code className="rounded bg-secondary px-2 py-1 font-mono text-sm">
                      {metrics.features[selectedFeature].formula}
                    </code>
                  </div>

                  <div>
                    <div className="mb-1 text-xs font-medium uppercase tracking-wider text-muted-foreground">
                      Interpretation
                    </div>
                    <p className="text-sm text-muted-foreground">
                      {metrics.features[selectedFeature].interpretation}
                    </p>
                  </div>

                  <Separator />

                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <div className="mb-1 text-xs font-medium uppercase tracking-wider text-muted-foreground">
                        Moyenne
                      </div>
                      <div className="font-mono text-lg">
                        {metrics.training.features.means[selectedFeature].toFixed(4)}
                      </div>
                    </div>
                    <div>
                      <div className="mb-1 text-xs font-medium uppercase tracking-wider text-muted-foreground">
                        Ecart-type
                      </div>
                      <div className="font-mono text-lg">
                        {metrics.training.features.stds[selectedFeature].toFixed(4)}
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </section>
        )}

        {/* Target Parameters */}
        {metrics.training && (
          <section className="mb-16">
            <div className="mb-6 flex items-center gap-2">
              <FontAwesomeIcon icon={faCircleInfo} className="h-4 w-4 text-muted-foreground" />
              <h2 className="text-lg font-medium">Parametres cibles</h2>
            </div>

            <div className="grid gap-6 sm:grid-cols-2">
              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="flex items-center gap-2 text-base">
                    <span className="font-mono">gamma</span>
                    <span className="text-muted-foreground">(γ)</span>
                  </CardTitle>
                  <CardDescription>
                    Parametre de phase du circuit QAOA. Controle l&apos;evolution sous le Hamiltonien du probleme.
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-3 gap-4 text-center">
                    <div>
                      <div className="text-xs text-muted-foreground">Min</div>
                      <div className="font-mono">{metrics.training.gamma.min.toFixed(3)}</div>
                    </div>
                    <div>
                      <div className="text-xs text-muted-foreground">Moyenne</div>
                      <div className="font-mono font-medium">{metrics.training.gamma.mean.toFixed(3)}</div>
                    </div>
                    <div>
                      <div className="text-xs text-muted-foreground">Max</div>
                      <div className="font-mono">{metrics.training.gamma.max.toFixed(3)}</div>
                    </div>
                  </div>
                  <div className="mt-4">
                    <div className="h-2 overflow-hidden rounded-full bg-secondary">
                      <div
                        className="h-full bg-primary"
                        style={{
                          marginLeft: `${((metrics.training.gamma.mean - metrics.training.gamma.min) /
                            (metrics.training.gamma.max - metrics.training.gamma.min)) * 100}%`,
                          width: "4px",
                        }}
                      />
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="flex items-center gap-2 text-base">
                    <span className="font-mono">beta</span>
                    <span className="text-muted-foreground">(β)</span>
                  </CardTitle>
                  <CardDescription>
                    Parametre de mixage du circuit QAOA. Controle l&apos;evolution sous le Hamiltonien de mixage.
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-3 gap-4 text-center">
                    <div>
                      <div className="text-xs text-muted-foreground">Min</div>
                      <div className="font-mono">{metrics.training.beta.min.toFixed(3)}</div>
                    </div>
                    <div>
                      <div className="text-xs text-muted-foreground">Moyenne</div>
                      <div className="font-mono font-medium">{metrics.training.beta.mean.toFixed(3)}</div>
                    </div>
                    <div>
                      <div className="text-xs text-muted-foreground">Max</div>
                      <div className="font-mono">{metrics.training.beta.max.toFixed(3)}</div>
                    </div>
                  </div>
                  <div className="mt-4">
                    <div className="h-2 overflow-hidden rounded-full bg-secondary">
                      <div
                        className="h-full bg-primary"
                        style={{
                          marginLeft: `${((metrics.training.beta.mean - metrics.training.beta.min) /
                            (metrics.training.beta.max - metrics.training.beta.min)) * 100}%`,
                          width: "4px",
                        }}
                      />
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </section>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-border">
        <div className="mx-auto max-w-5xl px-6 py-6">
          <div className="flex items-center justify-between text-sm text-muted-foreground">
            <span>FastQuantum - PoC Innovation</span>
            <span>Donnees reelles extraites du modele</span>
          </div>
        </div>
      </footer>
    </div>
  );
}

// Metric Card Component
function MetricCard({
  label,
  value,
  description,
}: {
  label: string;
  value: string;
  description: string;
}) {
  return (
    <Card>
      <CardContent className="pt-6">
        <div className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
          {label}
        </div>
        <div className="mt-1 font-mono text-2xl font-medium">{value}</div>
        <div className="mt-2 text-xs text-muted-foreground">{description}</div>
      </CardContent>
    </Card>
  );
}

// Dataset Card Component
function DatasetCard({ data }: { data: DatasetStats }) {
  return (
    <Card>
      <CardContent className="pt-6">
        <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
          <div>
            <div className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
              Echantillons
            </div>
            <div className="mt-1 font-mono text-xl">{data.n_samples.toLocaleString()}</div>
          </div>
          <div>
            <div className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
              Noeuds
            </div>
            <div className="mt-1 font-mono text-xl">
              {data.nodes.min} - {data.nodes.max}
            </div>
            <div className="text-xs text-muted-foreground">
              moy: {data.nodes.mean.toFixed(1)} (σ: {data.nodes.std.toFixed(1)})
            </div>
          </div>
          <div>
            <div className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
              Aretes
            </div>
            <div className="mt-1 font-mono text-xl">
              {data.edges.min} - {data.edges.max}
            </div>
            <div className="text-xs text-muted-foreground">
              moy: {data.edges.mean.toFixed(1)} (σ: {data.edges.std.toFixed(1)})
            </div>
          </div>
          <div>
            <div className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
              Features
            </div>
            <div className="mt-1 font-mono text-xl">7</div>
            <div className="text-xs text-muted-foreground">par noeud</div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
