const features = [
  { name: "Config format", trainformer: "Python", lightning: "Python", hf: "YAML + Python" },
  { name: "Vision tasks", trainformer: true, lightning: "manual", hf: "limited" },
  { name: "NLP tasks", trainformer: true, lightning: "manual", hf: true },
  { name: "Multimodal", trainformer: true, lightning: "manual", hf: "limited" },
  { name: "LoRA built-in", trainformer: true, lightning: false, hf: true },
  { name: "Learning curve", trainformer: "low", lightning: "medium", hf: "high" },
  { name: "Core lines", trainformer: "~600", lightning: "~5000+", hf: "~10000+" },
]

function Cell({ value }: { value: boolean | string }) {
  if (typeof value === "boolean") {
    return value ? (
      <span className="text-green-500">✓</span>
    ) : (
      <span className="text-red-500">✗</span>
    )
  }
  return <span className="text-muted">{value}</span>
}

export default function Comparison() {
  return (
    <section className="py-20">
      <div className="max-w-4xl mx-auto px-4">
        <h2 className="text-3xl font-bold text-foreground text-center mb-4">
          How does it compare?
        </h2>
        <p className="text-muted text-center mb-12 max-w-2xl mx-auto">
          trainformer is lean, focused, and covers all domains.
          No platform lock-in, no complex abstractions.
        </p>

        <div className="overflow-x-auto">
          <table className="w-full border-collapse">
            <thead>
              <tr className="border-b border-border">
                <th className="text-left py-4 px-4 text-foreground font-semibold">
                  Feature
                </th>
                <th className="text-center py-4 px-4 text-accent font-semibold">
                  trainformer
                </th>
                <th className="text-center py-4 px-4 text-muted font-medium">
                  Lightning
                </th>
                <th className="text-center py-4 px-4 text-muted font-medium">
                  HF Trainer
                </th>
              </tr>
            </thead>
            <tbody>
              {features.map((feature, index) => (
                <tr
                  key={feature.name}
                  className={index % 2 === 0 ? "bg-surface/30" : ""}
                >
                  <td className="py-3 px-4 text-foreground">{feature.name}</td>
                  <td className="py-3 px-4 text-center">
                    <Cell value={feature.trainformer} />
                  </td>
                  <td className="py-3 px-4 text-center">
                    <Cell value={feature.lightning} />
                  </td>
                  <td className="py-3 px-4 text-center">
                    <Cell value={feature.hf} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  )
}
