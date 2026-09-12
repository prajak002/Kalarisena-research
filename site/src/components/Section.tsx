export function Section({
  id,
  index,
  title,
  kicker,
  children,
}: {
  id: string;
  index: string;
  title: string;
  kicker?: string;
  children: React.ReactNode;
}) {
  return (
    <section id={id} className="section">
      <div className="section-head">
        <span className="section-index">{index}</span>
        <div>
          {kicker ? <div className="section-kicker">{kicker}</div> : null}
          <h2 className="section-title">{title}</h2>
        </div>
      </div>
      <div className="section-body">{children}</div>
    </section>
  );
}
