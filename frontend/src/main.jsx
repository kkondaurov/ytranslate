import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const POLL_INTERVAL_MS = 2500;

function formatTime(value) {
  if (!value) {
    return "-";
  }
  const date = new Date(value);
  if (Number.isNaN(date.valueOf())) {
    return value;
  }
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  }).format(date);
}

function valueOrDash(value) {
  return value === null || value === undefined || value === "" ? "-" : String(value);
}

function statusClass(value) {
  return valueOrDash(value).toLowerCase().replace(/[^a-z0-9_-]/g, "-");
}

function useStatus() {
  const [payload, setPayload] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    let timeoutId = null;

    async function refresh() {
      try {
        const response = await fetch("/api/status", { cache: "no-store" });
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        const nextPayload = await response.json();
        if (!cancelled) {
          setPayload(nextPayload);
          setError(null);
        }
      } catch (nextError) {
        if (!cancelled) {
          setError(nextError);
        }
      } finally {
        if (!cancelled) {
          timeoutId = window.setTimeout(refresh, POLL_INTERVAL_MS);
        }
      }
    }

    refresh();
    return () => {
      cancelled = true;
      if (timeoutId) {
        window.clearTimeout(timeoutId);
      }
    };
  }, []);

  return { payload, error };
}

function Badge({ value, tone }) {
  const className = tone ? statusClass(tone) : statusClass(value);
  return <span className={`badge ${className}`}>{valueOrDash(value)}</span>;
}

function Header({ payload, error }) {
  const jobs = payload?.jobs ?? [];
  const isLive = payload && !error;
  return (
    <header className="topbar">
      <div>
        <p className="eyebrow">Local pipeline</p>
        <h1>ytranslate status</h1>
      </div>
      <div className="status-strip" aria-live="polite">
        <span className="chip connection-chip">
          <span className={`dot ${isLive ? "connected" : error ? "disconnected" : ""}`} />
          <span>{isLive ? "Live" : error ? "Disconnected" : "Connecting"}</span>
        </span>
        <span className="chip">
          {error ? `Update failed: ${error.message}` : `Updated ${formatTime(payload?.generated_at)}`}
        </span>
        <span className="chip">{jobs.length} {jobs.length === 1 ? "job" : "jobs"}</span>
      </div>
    </header>
  );
}

function LatestJob({ job }) {
  if (!job) {
    return (
      <section className="panel latest-panel" aria-labelledby="latest-job-heading">
        <div className="panel-header">
          <h2 id="latest-job-heading">Latest job</h2>
          <Badge value="idle" />
        </div>
        <div className="panel-body empty-state">No jobs recorded.</div>
      </section>
    );
  }

  const title = job.title || job.title_translated || job.canonical_url || job.url || job.job_id;
  return (
    <section className="panel latest-panel" aria-labelledby="latest-job-heading">
      <div className="panel-header">
        <h2 id="latest-job-heading">Latest job</h2>
        <Badge value={job.status} />
      </div>
      <div className="panel-body">
        <div className="latest-main">
          <div className="latest-title">
            <h3>{title}</h3>
            {(job.canonical_url || job.url) && (
              <a className="url-line" href={job.canonical_url || job.url}>
                {job.canonical_url || job.url}
              </a>
            )}
          </div>
          <a className="badge neutral" href={`/jobs/${encodeURIComponent(job.job_id)}`}>JSON</a>
        </div>

        <dl className="detail-grid">
          <Detail label="Job" value={job.job_id} mono />
          <Detail label="Created" value={formatTime(job.created_at)} />
          <Detail label="Started" value={formatTime(job.started_at)} />
          <Detail label="Finished" value={formatTime(job.finished_at)} />
        </dl>

        {job.error && <div className="error-box">{job.error}</div>}
        {!!job.output_files?.length && (
          <div className="outputs">
            {job.output_files.map((path) => (
              <div className="output-row mono" key={path}>{path}</div>
            ))}
          </div>
        )}
      </div>
    </section>
  );
}

function Detail({ label, value, mono = false }) {
  return (
    <div className="detail">
      <dt>{label}</dt>
      <dd className={mono ? "mono" : undefined}>{valueOrDash(value)}</dd>
    </div>
  );
}

function PhasePanel({ job }) {
  const progress = job?.progress ?? {};
  const done = Number(progress.asr_chunks_done || 0);
  const total = Number(progress.asr_chunks_total || 0);
  const percent = job?.status === "succeeded"
    ? 100
    : total > 0
      ? Math.max(0, Math.min(100, Math.round((done / total) * 100)))
      : 0;

  return (
    <section className="panel" aria-labelledby="phase-heading">
      <div className="panel-header">
        <h2 id="phase-heading">Current phase</h2>
      </div>
      <div className="panel-body">
        <p className="phase-title">{valueOrDash(job?.phase_detail || job?.phase || "Idle")}</p>
        <p className="phase-detail">{job ? `${valueOrDash(job.status)} / ${valueOrDash(job.target_language)}` : "No active work."}</p>
        <div className="meter" aria-label={`Progress ${percent}%`}>
          <span style={{ width: `${percent}%` }} />
        </div>
      </div>
    </section>
  );
}

function StepRail({ steps = [] }) {
  const summary = useMemo(() => {
    const failed = steps.find((step) => step.state === "failed");
    const current = steps.find((step) => step.state === "current");
    const done = steps.filter((step) => step.state === "done").length;
    if (failed) {
      return `Failed at ${failed.label}`;
    }
    if (current) {
      return `Now: ${current.label}`;
    }
    if (steps.length) {
      return `${done} / ${steps.length} done`;
    }
    return "Idle";
  }, [steps]);

  return (
    <section className="panel" aria-labelledby="steps-heading">
      <div className="panel-header">
        <h2 id="steps-heading">Steps</h2>
        <span className="chip">{summary}</span>
      </div>
      <div className="panel-body">
        {steps.length ? (
          <ol className="steps-list">
            {steps.map((step, index) => (
              <li className={`step ${statusClass(step.state)}`} key={step.key ?? step.label}>
                <div className="step-top">
                  <span className="step-index">{index + 1}</span>
                  <Badge value={step.state} />
                </div>
                <div className="step-label">{valueOrDash(step.label)}</div>
                <div className="step-detail">{valueOrDash(step.detail)}</div>
              </li>
            ))}
          </ol>
        ) : (
          <p className="empty-state">No step data.</p>
        )}
      </div>
    </section>
  );
}

function EventsTable({ events = [] }) {
  const recent = events.slice(-40).reverse();
  return (
    <section className="panel" aria-labelledby="events-heading">
      <div className="panel-header">
        <h2 id="events-heading">Latest events</h2>
        <span className="chip">{recent.length} {recent.length === 1 ? "event" : "events"}</span>
      </div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr><th>Time</th><th>Level</th><th>Message</th></tr>
          </thead>
          <tbody>
            {recent.length ? recent.map((event) => (
              <tr key={`${event.at}-${event.message}`}>
                <td className="mono">{formatTime(event.at)}</td>
                <td className="event-level">{valueOrDash(event.level)}</td>
                <td className="event-message">{valueOrDash(event.message)}</td>
              </tr>
            )) : (
              <tr><td className="empty-table-cell" colSpan="3">No events recorded.</td></tr>
            )}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function RecentJobs({ jobs = [] }) {
  return (
    <section className="panel" aria-labelledby="recent-heading">
      <div className="panel-header">
        <h2 id="recent-heading">Recent jobs</h2>
      </div>
      <div className="table-wrap compact">
        <table>
          <thead>
            <tr><th>Created</th><th>Job</th><th>Status</th><th>Phase</th></tr>
          </thead>
          <tbody>
            {jobs.length ? jobs.map((job) => (
              <tr key={job.job_id}>
                <td className="mono">{formatTime(job.created_at)}</td>
                <td><a href={`/jobs/${encodeURIComponent(job.job_id)}`}>{job.job_id}</a></td>
                <td><Badge value={job.status} /></td>
                <td>{valueOrDash(job.phase_detail || job.phase)}</td>
              </tr>
            )) : (
              <tr><td className="empty-table-cell" colSpan="4">No jobs recorded.</td></tr>
            )}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function App() {
  const { payload, error } = useStatus();
  const latest = payload?.latest ?? null;
  const jobs = payload?.jobs ?? [];

  return (
    <div className="app">
      <Header payload={payload} error={error} />
      <main className="layout">
        <div className="left">
          <LatestJob job={latest} />
          <StepRail steps={latest?.steps ?? []} />
          <EventsTable events={latest?.events ?? []} />
        </div>
        <aside className="right">
          <PhasePanel job={latest} />
          <RecentJobs jobs={jobs} />
        </aside>
      </main>
    </div>
  );
}

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
