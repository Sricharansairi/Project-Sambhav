import React from 'react';
import { Shield, Brain, Activity, FileSearch, Eye, Database } from 'lucide-react';

export function PragmaDocumentation() {
  return (
    <div className="mt-8 p-6 rounded-2xl bg-[#08080c] border border-white/10 text-muted-foreground space-y-6">
      <div className="flex items-center gap-3 border-b border-white/10 pb-4">
        <Shield className="w-8 h-8 text-primary" />
        <div>
          <h2 className="text-xl font-bold text-foreground">PRAGMA: Forensic Psychological Profiling Engine</h2>
          <p className="text-xs uppercase tracking-widest text-primary/80 font-semibold">Technical Architecture & Methodology Guide</p>
        </div>
      </div>

      <div className="prose prose-invert prose-sm max-w-none text-xs leading-relaxed space-y-6">
        <section>
          <h3 className="text-sm font-semibold text-white flex items-center gap-2 mb-2"><Brain className="w-4 h-4 text-primary" /> 1. Introduction to PRAGMA</h3>
          <p>
            The PRAGMA (Pattern Recognition and Applied Geopolitical/Mental Analysis) engine represents the pinnacle of Project Sambhav’s multi-modal inference capabilities. Originally designed to interface with high-stakes adversarial negotiation environments, PRAGMA functions as a comprehensive forensic psychological profiler. Unlike traditional sentiment analysis engines that map binary emotional vectors (positive/negative), PRAGMA leverages a unified, state-of-the-art neuro-linguistic architecture to detect deception, cognitive load, underlying motives, and suppressed behavioral traits. 
            <br/><br/>
            Whether you are analyzing a witness statement, a corporate negotiation email, a political speech, or an interpersonal message, PRAGMA decomposes the semantic structure to reveal the raw psychological blueprint of the subject. It operates continuously in the background whenever the "Pragma" domain is selected, wrapping its probabilistic forecasts within an advanced psychological audit.
          </p>
        </section>

        <section>
          <h3 className="text-sm font-semibold text-white flex items-center gap-2 mb-2"><Activity className="w-4 h-4 text-primary" /> 2. Core Methodologies: The Deception Detection Matrix</h3>
          <p>
            PRAGMA approaches human deception through the lens of cognitive load-theory. Lying or obfuscating the truth requires significantly higher cognitive processing power than recounting a factual memory. The brain must simultaneously construct a plausible fabricated narrative, sequence it logically, monitor the recipient's reaction, and suppress the genuine memory. PRAGMA identifies this cognitive friction through four primary pillars:
          </p>
          <ul className="list-disc pl-5 mt-3 space-y-2">
            <li><strong>Linguistic Distancing:</strong> Deceptive individuals subconsciously separate themselves from their lies to reduce internal guilt. PRAGMA flags sudden drops in first-person pronouns ("I", "my") and an aggressive pivot toward passive voice or third-person generalities ("it was decided", "one might say").</li>
            <li><strong>Temporal Disassociation:</strong> When recounting true events, subjects naturally map events linearly using consistent past-tense structures. Fabricated accounts often feature "tense hopping"—a phenomenon where the subject accidentally slips into the present tense mid-story. PRAGMA’s temporal parser catches these micro-infractions instantly.</li>
            <li><strong>Over-Qualification & Hedging:</strong> Truth-tellers state facts plainly. Deceivers use buffer phrases ("to tell you the honest truth", "as far as I can recall", "essentially"). PRAGMA uses a calibrated LLM dictionary to assign adversarial weights to these markers.</li>
            <li><strong>Thematic Fragmentation:</strong> True accounts possess rich, spontaneous contextual details (environmental observations, minor irrelevant facts). Fabricated narratives are often sparse on spatial details but overly rigid regarding core events. PRAGMA's semantic density evaluator plots the distribution of peripheral vs. core memory clusters.</li>
          </ul>
        </section>

        <section>
          <h3 className="text-sm font-semibold text-white flex items-center gap-2 mb-2"><FileSearch className="w-4 h-4 text-primary" /> 3. The 8-Dimension Cognitive Audit</h3>
          <p>
            When processing a statement, PRAGMA does not simply output a "Lie/Truth" baseline. It executes an incredibly dense 8-dimensional cognitive audit, mapping probabilities across the following vectors:
          </p>
          <div className="grid grid-cols-2 gap-4 mt-3">
            <div className="p-3 bg-white/5 border border-white/5 rounded-lg">
              <strong className="text-white">A. Plausibility Physics:</strong> Evaluates if the sequence of events violates known physical or chronometric constraints.
            </div>
            <div className="p-3 bg-white/5 border border-white/5 rounded-lg">
              <strong className="text-white">B. Emotive Resonance:</strong> Matches the linguistic tone against the expected emotional gravity of the described event.
            </div>
            <div className="p-3 bg-white/5 border border-white/5 rounded-lg">
              <strong className="text-white">C. Evasiveness Index:</strong> Measures the frequency of deflected questions and non-answers.
            </div>
            <div className="p-3 bg-white/5 border border-white/5 rounded-lg">
              <strong className="text-white">D. Omission Signatures:</strong> Detects crucial "missing gaps" in time that the subject strategically skips over.
            </div>
            <div className="p-3 bg-white/5 border border-white/5 rounded-lg">
              <strong className="text-white">E. Machiavellian Drift:</strong> Scans for manipulative framing designed explicitly to invoke pity or anger in the listener.
            </div>
            <div className="p-3 bg-white/5 border border-white/5 rounded-lg">
              <strong className="text-white">F. Baseline Deviation:</strong> If previous parameters exist, measures how far the current statement deviates from the subject's normal syntax.
            </div>
          </div>
        </section>

        <section>
          <h3 className="text-sm font-semibold text-white flex items-center gap-2 mb-2"><Eye className="w-4 h-4 text-primary" /> 4. Actionable Intervention & Interview Strategy</h3>
          <p>
            PRAGMA surpasses traditional theoretical analysis by providing explicitly actionable intervention guidelines. If the system detects a 78% probability of timeline fabrication, the engine actively synthesizes an "Interview Strategy." 
            <br/><br/>
            By opening the <strong>PRAGMA Complete Analysis</strong> dialog box in the UI, investigators are provided with a live playbook: "The subject shows evasiveness around the 2:00 PM timeline. Recommend applying cognitive pressure by asking them to recount the events of the afternoon in reverse chronological order." This utilizes the psychological principle that fabricated memories are built linearly and collapse when forced into reverse sequence.
          </p>
        </section>

        <section>
          <h3 className="text-sm font-semibold text-white flex items-center gap-2 mb-2"><Database className="w-4 h-4 text-primary" /> 5. Deep Learning Architecture</h3>
          <p>
            The backend engine of PRAGMA relies on the Project Sambhav Route Engine. It initializes a "Devil's Advocate" context loop when generating the psychological profile. The primary LLM attempts to validate the truthfulness of the statement, while a secondary background agent acts adversarially, actively attempting to poke holes in the statement's logic.
            <br/><br/>
            The outputs of these two models are pushed through our proprietary <code>calculate_reliability</code> heuristic. If the confidence gap between the Truth agent and the Adversarial agent is heavily polarized, PRAGMA flags the statement with an "Anomalous Cognitive Warning," triggering the manual audit protocols seen in the interface.
            <br/><br/>
            <strong>Important Legal Disclaimer:</strong> The PRAGMA engine is a probabilistic analytical tool designed for research, strategy, and risk assessment parameters. It is not infallible. Cognitive indicators of stress (such as hedging and distancing) can also naturally occur in subjects suffering from trauma, anxiety disorders, or neurodivergence. Therefore, PRAGMA results must NEVER be used as unilateral verdicts in legal, terminal, or critical HR decisions without comprehensive independent human verification by certified forensic analysts.
          </p>
        </section>
      </div>
    </div>
  );
}
