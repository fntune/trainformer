const personas = [
  {
    icon: "🔬",
    title: "Researchers",
    description: "Running experiments that need to iterate fast and debug easily",
    quote: "I want to step through my training loop with a debugger",
  },
  {
    icon: "🛠️",
    title: "ML Engineers",
    description: "Building production pipelines without framework lock-in",
    quote: "I need to swap backbones without rewriting my pipeline",
  },
  {
    icon: "👥",
    title: "Teams",
    description: "Standardizing training across projects",
    quote: "I want my team to use consistent training patterns",
  },
]

export function WhoIsThisFor() {
  return (
    <section className="py-20 bg-surface/30">
      <div className="max-w-6xl mx-auto px-4">
        <h2 className="text-3xl font-bold text-foreground text-center mb-12">
          Who is trainformer for?
        </h2>

        <div className="grid md:grid-cols-3 gap-8">
          {personas.map((persona) => (
            <div
              key={persona.title}
              className="p-6 rounded-lg border border-border bg-surface/50 hover:border-accent/50 transition-colors"
            >
              <div className="text-4xl mb-4">{persona.icon}</div>
              <h3 className="text-xl font-semibold text-foreground mb-2">
                {persona.title}
              </h3>
              <p className="text-muted mb-4">{persona.description}</p>
              <p className="text-sm italic text-muted border-l-2 border-accent pl-4">
                "{persona.quote}"
              </p>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
