export default function Philosophy() {
  return (
    <section className="py-20 bg-surface/30">
      <div className="max-w-4xl mx-auto px-4">
        <blockquote className="text-center mb-12">
          <p className="text-2xl md:text-3xl font-semibold text-foreground mb-4">
            "Your experiment, your rules"
          </p>
          <p className="text-lg text-muted">
            trainformer gives you the iteration speed of notebooks with
            the code quality of production systems.
          </p>
        </blockquote>

        <div className="grid md:grid-cols-3 gap-8">
          <div className="text-center">
            <div className="w-12 h-12 rounded-full bg-accent/10 flex items-center justify-center mx-auto mb-4">
              <span className="text-accent text-xl font-bold">1</span>
            </div>
            <h3 className="text-lg font-semibold text-foreground mb-2">
              Debuggable by default
            </h3>
            <p className="text-sm text-muted">
              Step through any line, inspect any tensor
            </p>
          </div>

          <div className="text-center">
            <div className="w-12 h-12 rounded-full bg-accent/10 flex items-center justify-center mx-auto mb-4">
              <span className="text-accent text-xl font-bold">2</span>
            </div>
            <h3 className="text-lg font-semibold text-foreground mb-2">
              Pluggable architecture
            </h3>
            <p className="text-sm text-muted">
              Swap components without rewriting
            </p>
          </div>

          <div className="text-center">
            <div className="w-12 h-12 rounded-full bg-accent/10 flex items-center justify-center mx-auto mb-4">
              <span className="text-accent text-xl font-bold">3</span>
            </div>
            <h3 className="text-lg font-semibold text-foreground mb-2">
              Power when needed
            </h3>
            <p className="text-sm text-muted">
              Sensible defaults, full control available
            </p>
          </div>
        </div>
      </div>
    </section>
  )
}
