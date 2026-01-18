const features = [
  {
    title: "Mixed Precision",
    description: "fp16 / bf16",
    icon: "⚡",
  },
  {
    title: "torch.compile",
    description: "2x speedup",
    icon: "🚀",
  },
  {
    title: "Distributed",
    description: "via Accelerate",
    icon: "🌐",
  },
  {
    title: "Callbacks",
    description: "Checkpoint, EMA",
    icon: "🔔",
  },
  {
    title: "Logging",
    description: "WandB, TB, MLflow",
    icon: "📊",
  },
  {
    title: "Evaluation",
    description: "FAISS, KNN",
    icon: "📈",
  },
  {
    title: "Adapters",
    description: "LoRA, QLoRA",
    icon: "🔧",
  },
  {
    title: "Gradient Accumulation",
    description: "Train with any batch size",
    icon: "📦",
  },
  {
    title: "Export",
    description: "ONNX, TorchScript",
    icon: "📤",
  },
]

export default function Features() {
  return (
    <section className="py-20 bg-surface/30">
      <div className="max-w-6xl mx-auto px-4">
        <h2 className="text-3xl font-bold text-foreground text-center mb-4">
          Everything you need, nothing you don't
        </h2>
        <p className="text-muted text-center mb-12 max-w-2xl mx-auto">
          Production features when you need them, sensible defaults when you don't.
        </p>

        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {features.map((feature) => (
            <div
              key={feature.title}
              className="p-4 rounded-lg border border-border bg-background hover:border-accent/50 transition-colors"
            >
              <div className="flex items-start gap-3">
                <span className="text-2xl" aria-hidden="true">
                  {feature.icon}
                </span>
                <div>
                  <h3 className="font-semibold text-foreground">
                    {feature.title}
                  </h3>
                  <p className="text-sm text-muted">{feature.description}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
