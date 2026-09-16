import type { Grade } from "@/lib/types";

const NOTE: Record<Grade, string> = {
  A: "learned from data (184 Q/R sessions, 2022–25); interval from the model",
  B: "physics, coefficient checked on our data; interval from that check",
  C: "physics, literature value only; wide band, not verifiable with public data",
};
const CLS: Record<Grade, string> = {
  A: "text-grade-a bg-grade-abg border-grade-abd",
  B: "text-grade-b bg-grade-bbg border-grade-bbd",
  C: "text-grade-c bg-grade-cbg border-grade-cbd",
};

export function GradeChip({ grade }: { grade: Grade }) {
  return (
    <span title={`Grade ${grade}: ${NOTE[grade]}`}
      className={`inline-flex w-[18px] h-[18px] items-center justify-center rounded-[3px] border text-[10px] font-semibold font-mono ${CLS[grade]}`}>
      {grade}
    </span>
  );
}

export function GradeLegend() {
  return (
    <div className="text-[11px] text-hud-muted leading-relaxed flex flex-wrap gap-x-3 gap-y-1">
      {(["A", "B", "C"] as Grade[]).map((g) => (
        <span key={g} className="inline-flex items-center gap-1.5"><GradeChip grade={g} />{NOTE[g].split(";")[0]}</span>
      ))}
    </div>
  );
}
