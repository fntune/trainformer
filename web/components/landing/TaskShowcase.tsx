"use client"

import { Suspense } from "react"
import { Tabs } from "@/components/ui/Tabs"
import { CodeBlock } from "@/components/ui/CodeBlock"

const visionCode = `from trainformer.tasks import MetricLearning

task = MetricLearning(
    backbone="efficientnet_b0",
    loss="arcface",
    embedding_dim=512,
)
Trainer(task, data="stanford_products").fit()`

const nlpCode = `from trainformer.tasks import CausalLM
from trainformer.adapters import LoRA

task = CausalLM("meta-llama/Llama-2-7b", adapter=LoRA(r=8))
Trainer(task, data="databricks/dolly").fit()`

const multimodalCode = `from trainformer.tasks import VLM

task = VLM("llava-1.5-7b")
Trainer(task, data="coco_captions").fit()`

const tabs = [
  {
    id: "vision",
    label: "Vision",
    content: (
      <div className="space-y-4">
        <CodeBlock code={visionCode} lang="python" />
        <div className="flex flex-wrap gap-2">
          <span className="px-3 py-1 rounded-full bg-surface border border-border text-sm text-muted">
            ImageClassification
          </span>
          <span className="px-3 py-1 rounded-full bg-surface border border-border text-sm text-muted">
            MetricLearning (ArcFace, CosFace)
          </span>
          <span className="px-3 py-1 rounded-full bg-surface border border-border text-sm text-muted">
            SSL (SimCLR, MoCo, DINO, MAE)
          </span>
        </div>
      </div>
    ),
  },
  {
    id: "nlp",
    label: "NLP",
    content: (
      <div className="space-y-4">
        <CodeBlock code={nlpCode} lang="python" />
        <div className="flex flex-wrap gap-2">
          <span className="px-3 py-1 rounded-full bg-surface border border-border text-sm text-muted">
            CausalLM (GPT, Llama, Mistral)
          </span>
          <span className="px-3 py-1 rounded-full bg-surface border border-border text-sm text-muted">
            Seq2Seq (T5, BART)
          </span>
          <span className="px-3 py-1 rounded-full bg-surface border border-border text-sm text-muted">
            MaskedLM (BERT)
          </span>
        </div>
      </div>
    ),
  },
  {
    id: "multimodal",
    label: "Multimodal",
    content: (
      <div className="space-y-4">
        <CodeBlock code={multimodalCode} lang="python" />
        <div className="flex flex-wrap gap-2">
          <span className="px-3 py-1 rounded-full bg-surface border border-border text-sm text-muted">
            CLIP
          </span>
          <span className="px-3 py-1 rounded-full bg-surface border border-border text-sm text-muted">
            VLM (LLaVA, Qwen-VL)
          </span>
        </div>
      </div>
    ),
  },
]

function TaskShowcaseContent() {
  return (
    <section className="py-20 bg-surface/30">
      <div className="max-w-4xl mx-auto px-4">
        <h2 className="text-3xl font-bold text-foreground text-center mb-4">
          One interface for every domain
        </h2>
        <p className="text-muted text-center mb-12 max-w-2xl mx-auto">
          Vision, NLP, and Multimodal tasks share the same simple API.
          Switch domains without learning new frameworks.
        </p>

        <Tabs tabs={tabs} defaultTab="vision" ariaLabel="Task domains" />
      </div>
    </section>
  )
}

export default function TaskShowcase() {
  return (
    <Suspense fallback={<div className="py-20 text-center text-muted">Loading…</div>}>
      <TaskShowcaseContent />
    </Suspense>
  )
}
