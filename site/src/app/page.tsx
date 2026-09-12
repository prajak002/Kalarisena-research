import { Section } from "@/components/Section";
import { StepWalkthrough } from "@/components/StepWalkthrough";
import { Figure, VideoToggle } from "@/components/Media";
import { RetargetPipeline } from "@/components/RetargetPipeline";
import { SomaPipeline } from "@/components/SomaPipeline";
import { ArchitectureDiagram } from "@/components/ArchitectureDiagram";
import { Roadmap } from "@/components/Roadmap";
import { PushSweepChart } from "@/components/PushSweepChart";
import { InlineEq } from "@/components/Math";
import { StatusTag } from "@/components/StatusTag";
import {
  FAILURE_STEPS,
  PROJECTION_STEPS,
  VIABILITY_STEPS,
  CONTROL_STEPS,
  TRAINING_STAGES,
  CODE_TRACE,
  NOTATION,
  REPO_REAL_METRICS,
  PAPER_CLAIMED_METRICS,
  ABLATION_TABLE,
  ROADMAP_STAGES,
  CITATION,
} from "@/lib/content";
import { MetricGrid, StageTable, CodeTraceTable } from "@/components/DataTable";

export default function Home() {
  return (
    <>
      <nav className="topnav">
        <div className="topnav-inner">
          <span className="brand">KalariSena</span>
          <div className="nav-links">
            <a href="#problem">Problem</a>
            <a href="#retargeting">Retargeting</a>
            <a href="#method-projection">Method</a>
            <a href="#implementation">Implementation</a>
            <a href="#results">Results</a>
            <a href="#reproduce">Reproduce</a>
            <a href="/paper.pdf">Paper</a>
          </div>
        </div>
      </nav>

      <main className="shell">
        {/* HERO */}
        <header className="hero">
          <div className="hero-kicker">Interactive research walkthrough</div>
          <h1 className="hero-title">
            Feasible now does not mean viable for what comes next.
          </h1>
          <p className="hero-sub">
            KalariSena studies dynamic Kalaripayattu skill transfer to the Unitree G1
            humanoid: deep stances, rapid weight transfer, single-support balance, and
            momentum-heavy transitions. This page walks through the paper&rsquo;s method
            and, honestly, through what is and is not yet real in the accompanying
            repository: a physics stack with real measured results, a Stage A
            tracking policy trained for the first time (reward rose, stability did
            not fully follow), and a viability critic now trained against it for
            real (AUROC 0.931) - all still short of the paper&rsquo;s full multi-skill
            claim, and reported as such throughout.
          </p>
          <div className="hero-actions">
            <a className="btn btn-primary" href="#problem">Start the walkthrough</a>
            <a className="btn" href="/paper.pdf">Read the paper</a>
            <a className="btn" href="https://github.com/prajak002/Kalarisena-research">Code</a>
          </div>
          <video
            className="hero-video"
            src="/media/videos/retarget/overlay_kw_highkick_right.mp4"
            autoPlay
            loop
            muted
            playsInline
          />
          <p className="hero-video-caption">
            Real overlay from this repository&rsquo;s retargeting review: the human
            demonstration (translucent) and the Unitree G1 realization of a Kalaripayattu
            high kick, shown together - the same style of comparison the paper's
            Figure 1 uses.
          </p>
        </header>

        {/* 01 PROBLEM */}
        <Section id="problem" index="01" kicker="Research problem" title="The human-to-humanoid embodiment gap">
          <p className="lede">
            Kalaripayattu is built around tightly coupled posture, footwork, weight
            transfer, and whole-body momentum. A trajectory can be geometrically
            faithful to a human demonstration and still be physically unexecutable on a
            humanoid: different morphology, degrees of freedom, contact geometry, and
            actuation limits stand between a human motion and a robot that can actually
            perform it.
          </p>
          <p>
            The paper studies this specifically through Kalaripayattu because the art
            repeatedly forces four regimes that break naive retargeting: deep and
            asymmetric stances, rapid single-to-double support changes,
            momentum-intensive strikes and pivots, and temporally coupled transitions
            where whether the next phase is reachable depends on how the current one was
            executed.
          </p>
        </Section>

        {/* 02 RETARGETING PIPELINE */}
        <Section id="retargeting" index="02" kicker="Data pipeline" title="From video to a G1 reference: the real retargeting pipeline">
          <p className="lede">
            Before any control problem exists, a human demonstration has to become a
            candidate robot trajectory. This runs through <strong>GEM-X</strong> (NVlabs):
            <strong> SAM-3D-Body</strong> for 3D human reconstruction, <strong>SOMA</strong> for
            global motion normalization, and <strong>soma-retargeter</strong> for the
            human-to-G1 joint mapping. These three are cloned into the repository as
            dependencies and run as-is, not reimplemented. Below are that model chain&rsquo;s
            own real intermediate outputs for one clip, not an illustration of them.
          </p>
          <SomaPipeline />

          <p className="lede" style={{ marginTop: 36 }}>
            The retargeter&rsquo;s global 3D output above still has to become a specific
            robot trajectory, then be checked against the human source. These four
            stages are from this repository&rsquo;s own retargeting review tool
            (<code className="code-pill">kalarisena-review/</code>) for three different motions.
          </p>
          <RetargetPipeline />
          <p style={{ marginTop: 16, fontSize: 13 }}>
            Stage 2 (&ldquo;raw GEM-X retarget&rdquo;) is soma-retargeter&rsquo;s direct output onto
            the G1; stage 4 is after this repository&rsquo;s own
            {" "}<code className="code-pill">ground_correct_motions.py</code> pass fixes the
            floating-foot defect visible in stage 2.
          </p>
        </Section>

        {/* 03 FAILURE / MOTIVATION */}
        <Section id="motivation" index="03" kicker="Motivation" title="What happens without physics grounding">
          <p>
            Before presenting the method, the paper (and this repository&rsquo;s own
            experiment log) shows why it is needed. Step through the actual documented
            progression from raw retargeting to a stated research hypothesis.
          </p>
          <StepWalkthrough steps={FAILURE_STEPS} />
        </Section>

        {/* 04 KEY INSIGHT */}
        <Section id="insight" index="04" kicker="Key insight" title="Track, survive, preserve the intended future">
          <p className="callout">
            &ldquo;feasible at t &ne; viable for the intended future&rdquo;
          </p>
          <p>
            The paper positions this as a distinct question from three things prior
            work already studies well: whether a reference can be tracked, whether a
            controller is robust to disturbance, and whether a fallen robot can recover.
            None of those alone ask whether the <em>specific intended continuation</em> of
            a structured skill remains physically reachable from the current state.
          </p>
          <div className="chain-item">
            <span className="chain-label">imitation</span>
            <span><InlineEq tex="\text{track}" /> - can the reference be followed right now?</span>
          </div>
          <div className="chain-item">
            <span className="chain-label">robustness</span>
            <span><InlineEq tex="\text{survive}" /> - does the robot stay upright under disturbance?</span>
          </div>
          <div className="chain-item">
            <span className="chain-label">KalariSena</span>
            <span><InlineEq tex="\text{preserve the intended future}" /> - does a path to the designated next skill phase still exist?</span>
          </div>
        </Section>

        {/* 05 INTERACTIVE MAIN RESULT */}
        <Section id="main-result" index="05" kicker="Interactive result" title="A real, measured feasibility boundary">
          <p>
            This is the strongest result this repository can currently back with real
            data: a lateral push-recovery sweep on the horse stance, run against the
            repository&rsquo;s actual scripted PD + threshold-switch controller (not a
            learned policy - see the Implementation status section below). 36
            trials, 3 per force level, timing jittered by &plusmn;2 control frames.
          </p>
          <PushSweepChart />
          <div style={{ height: 20 }} />
          <VideoToggle
            options={[
              { label: "40N - recovers", src: "/media/videos/push_recovered_40N.mp4", caption: "40N lateral push: the horse stance absorbs it and holds." },
              { label: "120N - falls", src: "/media/videos/push_fallen_120N.mp4", caption: "120N lateral push: the same stance crosses the feasibility boundary and falls." },
            ]}
          />
          <p style={{ marginTop: 16 }}>
            The capture-point margin <InlineEq tex="\xi_t" /> separates the two outcomes
            perfectly across all 36 trials: about +0.05m on every recovery, about
            &minus;0.6m on every fall, with no partial-failure band at this resolution.
            This is exactly the kind of sharp, state-dependent feasibility boundary the
            paper argues a prospective viability estimate should anticipate before it is
            crossed - though here it is measured after the fact, from physics, not
            predicted in advance by a critic.
          </p>
        </Section>

        {/* 06 NOTATION */}
        <Section id="notation" index="06" kicker="Reference" title="Notation used throughout">
          <p>Symbols are used exactly as defined in the paper and kept consistent across every step below.</p>
          <div className="notation-grid">
            {NOTATION.map((n) => (
              <div className="notation-row" key={n.symbol}>
                <InlineEq tex={n.symbol} />
                <span style={{ color: "var(--fg-muted)", fontSize: 12.5 }}>{n.meaning}</span>
              </div>
            ))}
          </div>
        </Section>

        {/* 07 METHOD: PROJECTION */}
        <Section id="method-projection" index="07" kicker="Method, part 1" title="Physics-grounded embodiment projection">
          <p>
            The full pipeline, end to end. Colour marks what is real in the repository
            today versus what the paper proposes but does not yet exist in code.
          </p>
          <ArchitectureDiagram activeId="proj" />
          <p style={{ marginTop: 20 }}>
            Raw retargeting produces <InlineEq tex="Q^0_{1:T}" />: kinematically similar
            to the human demonstration, but not guaranteed executable. The projection
            stage refines it into <InlineEq tex="Q^*_{1:T}" />, a reference that keeps the
            movement&rsquo;s identity while satisfying contact, support, and dynamics
            constraints.
          </p>
          <StepWalkthrough steps={PROJECTION_STEPS} />
        </Section>

        {/* 08 METHOD: VIABILITY */}
        <Section id="method-viability" index="08" kicker="Method, part 2" title="Successor-conditioned viability">
          <ArchitectureDiagram activeId="scvc" />
          <p style={{ marginTop: 20 }}>
            The paper&rsquo;s central technical contribution: a critic that predicts,
            from the current state and the identity of a specific intended continuation,
            whether that continuation remains reachable - before tracking error
            reveals the failure.
          </p>
          <StepWalkthrough steps={VIABILITY_STEPS} />
        </Section>

        {/* 09 METHOD: CONTROL */}
        <Section id="method-control" index="09" kicker="Method, part 3" title="Intent-preserving corrective control">
          <ArchitectureDiagram activeId="residual" />
          <p style={{ marginTop: 20 }}>
            Viability estimates only matter if they change what the robot does. This
            stage converts <InlineEq tex="V_\psi" /> into the smallest correction that
            restores a path to the intended continuation, falling back to a conservative
            safety policy only once that path is judged unrecoverable.
          </p>
          <StepWalkthrough steps={CONTROL_STEPS} />
        </Section>

        {/* 10 IMPLEMENTATION TRACE */}
        <Section id="implementation" index="10" kicker="Implementation trace" title="From equation to code">
          <p>
            Every row below is checked against the repository directly. Where an
            equation has no corresponding file, that is stated rather than
            papered over.
          </p>
          <CodeTraceTable rows={CODE_TRACE} />
        </Section>

        {/* 11 TRAINING PIPELINE STATUS */}
        <Section id="training-status" index="11" kicker="Honesty check" title="Training pipeline: claimed vs. real">
          <p>
            The paper describes six PPO training stages (A through F). Here is what
            actually exists in the repository for each one, verified by reading every
            script rather than trusting file names.
          </p>
          <StageTable rows={TRAINING_STAGES} />
          <p style={{ marginTop: 16, fontSize: 13 }}>
            The repository&rsquo;s own asset map originally stated:
            {" "}<code className="code-pill">results_paper/PAPER_ASSETS.md</code> -
            &ldquo;Stage A&ndash;F learned policies do not exist; the train_*.py files are
            stubs.&rdquo; That was true when written. Stage A has since been trained for
            real - see below.
          </p>
        </Section>

        {/* 11.1 REAL TRAINED RESULT */}
        <Section id="trained-result" index="11.1" kicker="Real result" title="Stage A, trained: what it actually looks like">
          <p>
            Reward climbed throughout training, but a real evaluation (5 episodes,
            deterministic policy) tells a more honest story than the training curve
            alone:
          </p>
          <div className="table-wrap">
            <table className="data-table">
              <thead><tr><th></th><th>Trained PPO policy</th><th>Raw PD baseline (no RL)</th></tr></thead>
              <tbody>
                <tr><td>Fall rate</td><td className="mono">100% (5/5)</td><td className="mono">100% (5/5)</td></tr>
                <tr><td>Mean episode length</td><td className="mono">85 steps</td><td className="mono">37 steps</td></tr>
                <tr><td>Tracking RMSE</td><td className="mono">0.309</td><td className="mono">0.258 (better)</td></tr>
              </tbody>
            </table>
          </div>
          <div className="figure-grid" style={{ marginTop: 16 }}>
            <div>
              <video className="video-frame" src="/media/videos/training/eval_policy.mp4" controls muted loop playsInline />
              <p className="figure-caption">Trained policy.</p>
            </div>
            <div>
              <video className="video-frame" src="/media/videos/training/eval_pd_baseline.mp4" controls muted loop playsInline />
              <p className="figure-caption">Raw PD baseline.</p>
            </div>
          </div>
          <p style={{ marginTop: 16 }}>
            The trained policy survives about 2.3x longer before falling, but still
            falls in every evaluation episode, and its raw tracking accuracy is
            slightly worse than doing nothing extra at all. 3,000,000 steps of
            tracking-only reward on one motion, with no CoM/balance shaping (Stage B,
            still a stub) and no viability-gated correction, does not solve stability
            here - exactly the gap the paper&rsquo;s method is designed to close.
          </p>
          <p className="lede" style={{ marginTop: 24 }}>Post-training behaviour under a lateral push (same trained policy):</p>
          <div className="figure-grid" style={{ gridTemplateColumns: "1fr 1fr 1fr" }}>
            <div>
              <video className="video-frame" src="/media/videos/training/thrust_trained_20N.mp4" controls muted loop playsInline />
              <p className="figure-caption">20N: fell (34 steps).</p>
            </div>
            <div>
              <video className="video-frame" src="/media/videos/training/thrust_trained_60N.mp4" controls muted loop playsInline />
              <p className="figure-caption">60N: recovered (99 steps).</p>
            </div>
            <div>
              <video className="video-frame" src="/media/videos/training/thrust_trained_100N.mp4" controls muted loop playsInline />
              <p className="figure-caption">100N: fell (92 steps).</p>
            </div>
          </div>
          <p style={{ marginTop: 12, fontSize: 13 }}>
            Non-monotonic (survives 60N but not the smaller 20N push) - reported as
            measured, not smoothed over.
          </p>
        </Section>

        {/* 12 QUALITATIVE RESULTS */}
        <Section id="qualitative" index="12" kicker="Qualitative" title="Motion corpus and physical challenges">
          <Figure
            src="/media/figures/fig1_motion_corpus.png"
            caption="Real curated clips from the motion corpus, family-labelled: stance, footwork, weight transfer, single-leg, pivot/rotation, explosive advance, jump/landing, body sequence."
            width={1600}
            height={500}
          />
          <Figure
            src="/media/figures/fig2_physical_challenges.png"
            caption="Real MuJoCo states with CoM / support / capture-point overlays computed by the same Pinocchio path the controller uses (cross-engine CoM agreement 1.04e-6m)."
            width={1600}
            height={500}
          />
        </Section>

        {/* 13 QUANTITATIVE RESULTS */}
        <Section id="results" index="13" kicker="Quantitative" title="Numbers: measured here vs. claimed in the paper">
          <p className="lede">Measured in this repository (scripted controller):</p>
          <MetricGrid metrics={REPO_REAL_METRICS} />
          <p className="lede" style={{ marginTop: 32 }}>As reported in the paper (not yet reproduced here):</p>
          <MetricGrid metrics={PAPER_CLAIMED_METRICS} />

          <p style={{ marginTop: 32 }}>Paper&rsquo;s component ablation (Table 5b), reported as-is:</p>
          <div className="table-wrap">
            <table className="data-table">
              <thead>
                <tr><th>Variant</th><th>Intent Preservation Rate</th></tr>
              </thead>
              <tbody>
                {ABLATION_TABLE.map((r) => (
                  <tr key={r.variant}><td>{r.variant}</td><td className="mono">{r.ipr}</td></tr>
                ))}
              </tbody>
            </table>
          </div>
        </Section>

        {/* 14 FAILURE CASES */}
        <Section id="limitations" index="14" kicker="Limitations" title="What is honestly unsolved">
          <ul style={{ color: "var(--fg-muted)", paddingLeft: 20, display: "flex", flexDirection: "column", gap: 10 }}>
            <li>Single-support balance is unsolved in the repository: a 36-pose grid search found no static single-leg stance surviving 2.5s under position-PD tracking.</li>
            <li>A hand-designed CoM-feedback (ankle/hip) balance controller was implemented and failed at every gain tested - kept in the tree, disabled, as a documented negative result.</li>
            <li>The fall response in the repository is a scripted protective crouch, not a learned recovery policy.</li>
            <li>None of the paper&rsquo;s twelve external baselines (KungfuBot, SONIC, BeyondMimic, Switch, SafeFlow, and others) are reproduced in this repository yet.</li>
            <li>Physical hardware deployment (paper Table 6) has no counterpart here - every result on this page is simulation-only.</li>
          </ul>
        </Section>

        {/* 15 ROADMAP */}
        <Section id="roadmap" index="15" kicker="Beyond the paper" title="Repo roadmap: Stages G, H, I">
          <p>
            Separate from the paper, the repository has an internal engineering roadmap
            (<code className="code-pill">docs/PHYSICAL_AI_STAGE_G_H_I_DRAFT.md</code>) for extending the
            existing Stage A&ndash;F pipeline. None of this is implemented and none of it
            is a paper claim - it is real, current planning, included here as-is.
          </p>
          <Roadmap stages={ROADMAP_STAGES} />
        </Section>

        {/* 16 REPRODUCE */}
        <Section id="reproduce" index="16" kicker="Reproducibility" title="Reproduce what is real">
          <p>Every command below reruns something that actually exists and writes real output, not typed-in numbers.</p>
          <pre className="citation-block">{`# Stage 1 gate: cross-engine physics validation
python3 code/scripts/test_sim_stage1.py

# Push-recovery sweep (the chart above)
python3 code/scripts/sim_push_sweep.py --out results_sim/

# Fall-severity A/B
python3 code/scripts/sim_fall_ab.py --out results_sim/

# Rebuild the honest paper-style tables from real trial CSVs
python3 code/scripts/make_tables.py --results results_sim --out results_paper`}</pre>
        </Section>

        {/* 17 LINKS */}
        <Section id="links" index="17" kicker="Resources" title="Code, paper, dataset">
          <div className="hero-actions">
            <a className="btn" href="/paper.pdf">Paper (PDF)</a>
            <a className="btn" href="https://github.com/prajak002/Kalarisena-research">Repository</a>
            <a className="btn" href="/code/results_paper">Result tables and figures</a>
          </div>
        </Section>

        {/* 18 CITATION */}
        <Section id="citation" index="18" kicker="Citation" title="Cite this work">
          <pre className="citation-block">{CITATION}</pre>
        </Section>

        <footer className="site-footer">
          <p>
            Built as an interactive companion to the KalariSena paper. Status tags on
            this page (
            <StatusTag status="repo-real" />, <StatusTag status="paper" />,{" "}
            <StatusTag status="repo-stub" />, <StatusTag status="negative" />
            ) reflect a direct audit of the accompanying repository, not the paper&rsquo;s
            claims alone.
          </p>
        </footer>
      </main>
    </>
  );
}
