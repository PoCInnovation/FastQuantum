"use client";

import { useState, useEffect } from "react";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import {
  faAtom,
  faBrain,
  faChartLine,
  faDatabase,
  faArrowRight,
  faPlay,
  faWaveSquare,
} from "@fortawesome/free-solid-svg-icons";
import { faGithub } from "@fortawesome/free-brands-svg-icons";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs";
import { Separator } from "@/components/ui/separator";
import Link from "next/link";

// Types
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
        <div className="flex flex-col items-center gap-3">
          <div className="h-8 w-8 rounded-full border-2 border-muted border-t-foreground animate-spin" />
          <span className="text-sm text-muted-foreground">Chargement...</span>
        </div>
      </div>
    );
  }

  if (!metrics) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <div className="max-w-sm text-center">
          <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-2xl bg-muted">
            <FontAwesomeIcon icon={faAtom} className="h-5 w-5 text-muted-foreground" />
          </div>
          <p className="text-sm text-muted-foreground leading-relaxed">
            Lancez{" "}
            <code className="rounded-md bg-muted px-2 py-0.5 font-mono text-xs text-foreground">
              python scripts/extract_metrics.py
            </code>{" "}
            pour generer les metriques.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Navigation */}
      <header className="sticky top-0 z-50 bg-white/80 backdrop-blur-xl border-b border-border/60">
        <div className="mx-auto flex max-w-5xl items-center justify-between px-6 py-4">
          <Link href="/" className="flex items-center gap-2.5">
            <div className="flex h-8 w-8 items-center justify-center rounded-xl bg-foreground">
              <FontAwesomeIcon icon={faAtom} className="h-3.5 w-3.5 text-white" />
            </div>
            <span className="text-[15px] font-bold tracking-tight">FastQuantum</span>
          </Link>
          <nav className="flex items-center gap-5">
            <Link
              href="/playground"
              className="group flex items-center gap-1.5 text-[13px] font-medium text-muted-foreground transition-colors hover:text-foreground"
            >
              Playground
              <FontAwesomeIcon
                icon={faArrowRight}
                className="h-2.5 w-2.5 opacity-0 -translate-x-1 transition-all group-hover:opacity-100 group-hover:translate-x-0"
              />
            </Link>
            <a
              href="https://github.com/PoCInnovation/FastQuantum"
              target="_blank"
              rel="noopener noreferrer"
              className="text-muted-foreground transition-colors hover:text-foreground"
            >
              <FontAwesomeIcon icon={faGithub} className="h-[18px] w-[18px]" />
            </a>
          </nav>
        </div>
      </header>

      {/* Hero */}
      <section className="relative overflow-hidden border-b border-border/40">
        {/* Subtle gradient orbs */}
        <div className="absolute -top-40 -right-40 h-80 w-80 rounded-full bg-indigo-50 opacity-60 blur-3xl" />
        <div className="absolute -bottom-20 -left-20 h-60 w-60 rounded-full bg-violet-50 opacity-40 blur-3xl" />

        <div className="relative mx-auto max-w-5xl px-6 pt-20 pb-16">
          <div className="animate-fade-up">
            <div className="mb-5 inline-flex items-center gap-2 rounded-full border border-border/80 bg-muted/50 px-3 py-1">
              <div className="h-1.5 w-1.5 rounded-full bg-accent animate-pulse" />
              <span className="text-[11px] font-semibold uppercase tracking-widest text-muted-foreground">
                GNN + QAOA
              </span>
            </div>
          </div>

          <h1 className="animate-fade-up delay-75 text-[clamp(2.25rem,5vw,3.5rem)] font-extrabold leading-[1.08] tracking-tight text-foreground">
            Prediction des parametres
            <br />
            <span className="text-accent">QAOA</span>
          </h1>

          <p className="animate-fade-up delay-150 mt-5 max-w-lg text-[15px] leading-relaxed text-muted-foreground">
            Utilisation de Graph Neural Networks pour predire les parametres
            optimaux <span className="font-mono text-foreground/80">γ</span> et{" "}
            <span className="font-mono text-foreground/80">β</span>,
            accelerant l&apos;optimisation de circuits quantiques.
          </p>

          <div className="animate-fade-up delay-200 mt-8 flex items-center gap-3">
            <Link href="/playground">
              <Button size="lg" className="h-11 gap-2 rounded-xl px-6 font-semibold shadow-sm">
                <FontAwesomeIcon icon={faPlay} className="h-3 w-3" />
                Tester le modele
              </Button>
            </Link>
            <a
              href="https://github.com/PoCInnovation/FastQuantum"
              target="_blank"
              rel="noopener noreferrer"
            >
              <Button
                variant="outline"
                size="lg"
                className="h-11 gap-2 rounded-xl px-6 font-semibold"
              >
                <FontAwesomeIcon icon={faGithub} className="h-4 w-4" />
                Source
              </Button>
            </a>
          </div>
        </div>
      </section>

      <main className="mx-auto max-w-5xl px-6 py-16">
        {/* Model Performance */}
        {metrics.model && (
          <section className="mb-20 animate-fade-up delay-300">
            <SectionHeader
              icon={faBrain}
              title="Performance du modele"
              badge={metrics.model.model_type}
            />
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              <MetricCard
                label="Validation Loss"
                value={metrics.model.val_loss.toFixed(6)}
                sub="MSE sur validation"
              />
              <MetricCard
                label="Gamma MAE"
                value={metrics.model.gamma_mae.toFixed(4)}
                sub="Erreur absolue (γ)"
                accent
              />
              <MetricCard
                label="Beta MAE"
                value={metrics.model.beta_mae.toFixed(4)}
                sub="Erreur absolue (β)"
                accent
              />
              <MetricCard
                label="Parametres"
                value={metrics.model.total_parameters.toLocaleString()}
                sub={`${metrics.model.epoch} epochs`}
              />
            </div>
          </section>
        )}

        {/* Dataset */}
        {metrics.training && metrics.validation && (
          <section className="mb-20">
            <SectionHeader icon={faDatabase} title="Donnees d'entrainement" />

            <Tabs defaultValue="training" className="w-full">
              <TabsList className="mb-4 h-10 bg-muted/50 p-1 rounded-xl">
                <TabsTrigger
                  value="training"
                  className="rounded-lg text-[13px] data-[state=active]:bg-white data-[state=active]:shadow-sm"
                >
                  Entrainement ({metrics.training.n_samples})
                </TabsTrigger>
                <TabsTrigger
                  value="validation"
                  className="rounded-lg text-[13px] data-[state=active]:bg-white data-[state=active]:shadow-sm"
                >
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

        {/* Features */}
        {metrics.features && metrics.training && (
          <section className="mb-20">
            <SectionHeader
              icon={faChartLine}
              title="Features des noeuds"
              subtitle="7 par noeud"
            />

            <div className="grid gap-5 lg:grid-cols-[240px_1fr]">
              <div className="flex flex-col gap-0.5">
                {metrics.features.map((feature, i) => (
                  <button
                    key={feature.key}
                    onClick={() => setSelectedFeature(i)}
                    className={`group flex items-center justify-between rounded-xl px-3.5 py-2.5 text-left text-[13px] transition-all ${
                      selectedFeature === i
                        ? "bg-foreground text-white font-semibold shadow-sm"
                        : "text-muted-foreground hover:text-foreground hover:bg-muted/70"
                    }`}
                  >
                    <span>{feature.name}</span>
                    <FontAwesomeIcon
                      icon={faArrowRight}
                      className={`h-2.5 w-2.5 transition-all ${
                        selectedFeature === i
                          ? "opacity-60"
                          : "opacity-0 group-hover:opacity-30"
                      }`}
                    />
                  </button>
                ))}
              </div>

              <div className="premium-card rounded-2xl p-6">
                <div className="mb-1 text-[15px] font-bold text-foreground">
                  {metrics.features[selectedFeature].name}
                </div>
                <p className="mb-5 text-[13px] text-muted-foreground">
                  {metrics.features[selectedFeature].description}
                </p>

                <div className="space-y-5">
                  <div>
                    <div className="mb-1.5 text-[11px] font-semibold uppercase tracking-widest text-muted-foreground/60">
                      Formule
                    </div>
                    <code className="inline-block rounded-lg bg-muted/70 px-3 py-1.5 font-mono text-[13px] text-foreground">
                      {metrics.features[selectedFeature].formula}
                    </code>
                  </div>

                  <div>
                    <div className="mb-1.5 text-[11px] font-semibold uppercase tracking-widest text-muted-foreground/60">
                      Interpretation
                    </div>
                    <p className="text-[13px] leading-relaxed text-muted-foreground">
                      {metrics.features[selectedFeature].interpretation}
                    </p>
                  </div>

                  <div className="divider-fade" />

                  <div className="grid grid-cols-2 gap-6">
                    <div>
                      <div className="mb-1 text-[11px] font-semibold uppercase tracking-widest text-muted-foreground/60">
                        Moyenne
                      </div>
                      <div className="font-mono text-xl font-bold text-foreground">
                        {metrics.training.features.means[selectedFeature].toFixed(4)}
                      </div>
                    </div>
                    <div>
                      <div className="mb-1 text-[11px] font-semibold uppercase tracking-widest text-muted-foreground/60">
                        Ecart-type
                      </div>
                      <div className="font-mono text-xl font-bold text-accent">
                        {metrics.training.features.stds[selectedFeature].toFixed(4)}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </section>
        )}

        {/* Target Parameters */}
        {metrics.training && (
          <section className="mb-16">
            <SectionHeader icon={faWaveSquare} title="Parametres cibles" />

            <div className="grid gap-4 sm:grid-cols-2">
              <ParameterCard
                name="gamma"
                symbol="γ"
                description="Parametre de phase — controle l'evolution sous le Hamiltonien du probleme."
                data={metrics.training.gamma}
                accent
              />
              <ParameterCard
                name="beta"
                symbol="β"
                description="Parametre de mixage — controle l'evolution sous le Hamiltonien de mixage."
                data={metrics.training.beta}
              />
            </div>
          </section>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-border/40">
        <div className="mx-auto max-w-5xl px-6 py-8">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-[13px] text-muted-foreground">
              <div className="flex h-5 w-5 items-center justify-center rounded-md bg-foreground">
                <FontAwesomeIcon icon={faAtom} className="h-2.5 w-2.5 text-white" />
              </div>
              FastQuantum — PoC Innovation
            </div>
            <span className="text-[12px] text-muted-foreground/60">
              Donnees reelles du modele
            </span>
          </div>
        </div>
      </footer>
    </div>
  );
}

/* ===== Sub-components ===== */

function SectionHeader({
  icon,
  title,
  badge,
  subtitle,
}: {
  icon: typeof faBrain;
  title: string;
  badge?: string;
  subtitle?: string;
}) {
  return (
    <div className="mb-6 flex items-center gap-2.5">
      <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-muted">
        <FontAwesomeIcon icon={icon} className="h-3 w-3 text-muted-foreground" />
      </div>
      <h2 className="text-[15px] font-bold tracking-tight">{title}</h2>
      {badge && (
        <Badge variant="secondary" className="ml-1 text-[11px] font-mono font-medium rounded-md">
          {badge}
        </Badge>
      )}
      {subtitle && (
        <span className="text-[12px] text-muted-foreground">{subtitle}</span>
      )}
    </div>
  );
}

function MetricCard({
  label,
  value,
  sub,
  accent,
}: {
  label: string;
  value: string;
  sub: string;
  accent?: boolean;
}) {
  return (
    <div className="premium-card rounded-2xl p-5">
      <div className="text-[11px] font-semibold uppercase tracking-widest text-muted-foreground/60">
        {label}
      </div>
      <div
        className={`mt-2 font-mono text-2xl font-bold tracking-tight ${
          accent ? "text-accent" : "text-foreground"
        }`}
      >
        {value}
      </div>
      <div className="mt-1.5 text-[12px] text-muted-foreground">{sub}</div>
    </div>
  );
}

function DatasetCard({ data }: { data: DatasetStats }) {
  return (
    <div className="premium-card rounded-2xl p-6">
      <div className="grid gap-8 sm:grid-cols-2 lg:grid-cols-4">
        <StatItem label="Echantillons" value={data.n_samples.toLocaleString()} />
        <StatItem
          label="Noeuds"
          value={`${data.nodes.min} — ${data.nodes.max}`}
          detail={`μ ${data.nodes.mean.toFixed(1)}  σ ${data.nodes.std.toFixed(1)}`}
        />
        <StatItem
          label="Aretes"
          value={`${data.edges.min} — ${data.edges.max}`}
          detail={`μ ${data.edges.mean.toFixed(1)}  σ ${data.edges.std.toFixed(1)}`}
        />
        <StatItem label="Features" value="7" detail="par noeud" />
      </div>
    </div>
  );
}

function StatItem({
  label,
  value,
  detail,
}: {
  label: string;
  value: string;
  detail?: string;
}) {
  return (
    <div>
      <div className="text-[11px] font-semibold uppercase tracking-widest text-muted-foreground/60">
        {label}
      </div>
      <div className="mt-1 font-mono text-lg font-bold">{value}</div>
      {detail && (
        <div className="mt-0.5 font-mono text-[11px] text-muted-foreground">{detail}</div>
      )}
    </div>
  );
}

function ParameterCard({
  name,
  symbol,
  description,
  data,
  accent,
}: {
  name: string;
  symbol: string;
  description: string;
  data: { min: number; max: number; mean: number; std: number };
  accent?: boolean;
}) {
  const meanPercent = ((data.mean - data.min) / (data.max - data.min)) * 100;
  const barColor = accent ? "bg-accent" : "bg-foreground";
  const symbolColor = accent ? "text-accent" : "text-foreground";

  return (
    <div className="premium-card rounded-2xl p-6">
      <div className="flex items-center gap-3 mb-3">
        <div className={`flex h-9 w-9 items-center justify-center rounded-xl ${accent ? "bg-accent/8" : "bg-muted"}`}>
          <span className={`font-mono text-base font-bold ${symbolColor}`}>{symbol}</span>
        </div>
        <div>
          <div className="font-mono text-[15px] font-bold">{name}</div>
        </div>
      </div>
      <p className="mb-5 text-[13px] leading-relaxed text-muted-foreground">{description}</p>

      <div className="grid grid-cols-3 gap-3 text-center mb-4">
        <div className="rounded-xl bg-muted/50 px-2 py-2.5">
          <div className="text-[10px] font-semibold uppercase tracking-widest text-muted-foreground/50">Min</div>
          <div className="font-mono text-[13px] font-bold mt-0.5">{data.min.toFixed(3)}</div>
        </div>
        <div className={`rounded-xl px-2 py-2.5 ${accent ? "bg-accent/5" : "bg-muted/50"}`}>
          <div className="text-[10px] font-semibold uppercase tracking-widest text-muted-foreground/50">Moyenne</div>
          <div className={`font-mono text-[13px] font-bold mt-0.5 ${symbolColor}`}>{data.mean.toFixed(3)}</div>
        </div>
        <div className="rounded-xl bg-muted/50 px-2 py-2.5">
          <div className="text-[10px] font-semibold uppercase tracking-widest text-muted-foreground/50">Max</div>
          <div className="font-mono text-[13px] font-bold mt-0.5">{data.max.toFixed(3)}</div>
        </div>
      </div>

      <div className="relative">
        <div className="h-1 rounded-full bg-muted">
          <div
            className={`h-full rounded-full ${barColor} transition-all duration-500`}
            style={{ width: `${meanPercent}%` }}
          />
        </div>
      </div>
    </div>
  );
}
