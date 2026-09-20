#!/usr/bin/env python3
"""Regenerate the paper-facing reproduction catalog and local configurator.

Source of truth: reproduction_utils.catalog.TASKS + PAPER_INFO.
The generated HTML is self-contained and works when opened directly from a
local clone (file://); it never writes to reproduction_config.json.
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reproduction_utils.catalog import PAPER_INFO, TASKS  # noqa: E402
from reproduction_utils.model_variants import task_variant_ids  # noqa: E402

STAR_LABEL = {
    None: "",
    "mu2": "★μ2",
    "gpic": "★GPIC",
    "rn_surface": "★RNsurf",
}
TYPE_LABEL = {"M": "[M]", "I": "[I]", "M/I": "[M/I]"}


def catalog_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, task in enumerate(TASKS, 1):
        info = PAPER_INFO[task.id]
        rows.append(
            {
                "number": index,
                "id": task.id,
                "title": task.title,
                "family": task.family,
                "tier": task.tier,
                "kind": task.kind,
                "deps": list(task.deps),
                "selectable": not task.manual_args and task.tier != "utility",
                "models": list(task_variant_ids(task.id)),
                "experiment_type": info.experiment_type,
                "type_label": TYPE_LABEL[info.experiment_type],
                "intervention_family": info.intervention_family,
                "star_label": STAR_LABEL[info.intervention_family],
                "paper_sections": list(info.paper_sections),
                "paper_artifacts": list(info.paper_artifacts),
                "description": info.description or task.description or task.title,
                "thumbnail": info.thumbnail,
                "special_tags": list(info.special_tags),
            }
        )
    return rows


def render_readme(rows: list[dict[str, object]]) -> str:
    lines = [
        "# Paper experiment catalog",
        "",
        "This is the visual index for `reproduce.py`. Running numbers are for recognition only; ",
        "the stable machine identifiers are the task IDs. For choosing a subset, open ",
        "[`configurator.html`](configurator.html) from your local clone and download/copy a ",
        "selection JSON.",
        "",
        "**Selection files contain task IDs only.** They do not replace or merge over ",
        "`reproduction_config.json`, so dataset paths, model paths, device settings, and the ",
        "output root configured by `python reproduce.py setup` remain authoritative.",
        "",
        "Legend: `[M]` measurement/observation, `[I]` explicit intervention, `[M/I]` both. ",
        "`★μ2` marks CLS/register μ2 steering, `★GPIC` Conv1 manifold surfing, and ",
        "`★RNsurf` RN-induced control-surface experiments.",
        "",
    ]

    families: list[str] = []
    for row in rows:
        fam = str(row["family"])
        if fam not in families:
            families.append(fam)

    for family in families:
        fam_rows = [r for r in rows if r["family"] == family]
        title = {
            "bridge": "Bridge / trained-model behavior",
            "workspace": "Native workspace / register circuit",
            "rn": "RN mechanism / universality",
            "figures": "Figure/postprocess tasks",
            "conv1": "Conv1 / early routing",
            "audit": "Utility / audit",
        }.get(family, family)
        lines += [f"## {title}", "", "| # | Preview | Experiment | Type | Paper anchor(s) | What it does |", "|---:|---|---|---|---|---|"]
        for row in fam_rows:
            thumb = row["thumbnail"]
            if thumb:
                preview = f'<img src="figures_thumbs/{html.escape(str(thumb))}" width="250" alt="{html.escape(str(row["title"]))}">' 
            else:
                preview = "—"
            paper = "<br>".join(f"`{html.escape(str(x))}`" for x in row["paper_sections"]) or "support / utility"
            type_text = f'`{row["type_label"]}`'
            if row["star_label"]:
                type_text += f' {row["star_label"]}'
            if row["special_tags"]:
                type_text += " " + " ".join(f"({tag})" for tag in row["special_tags"])
            select_note = "" if row["selectable"] else "<br><sub>manual utility</sub>"
            lines.append(
                f'| {row["number"]} | {preview} | **`{row["id"]}`**<br>{html.escape(str(row["title"]))}{select_note} | '
                f'{type_text} | {paper} | {html.escape(str(row["description"]))} |'
            )
        lines.append("")

    return "\n".join(lines)


def render_html(rows: list[dict[str, object]]) -> str:
    payload = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    template = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ModeMUX paper experiment configurator</title>
<style>
:root { color-scheme: light dark; --bg:#101114; --panel:#17191f; --muted:#aab0be; --line:#303542; --accent:#82b7ff; --good:#8bd49c; }
* { box-sizing:border-box; }
body { margin:0; font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif; background:var(--bg); color:#f2f4f8; }
a { color:var(--accent); }
header { padding:24px clamp(16px,4vw,48px) 12px; max-width:1500px; margin:auto; }
header h1 { margin:0 0 8px; }
header p { max-width:1000px; color:var(--muted); line-height:1.5; }
.warning { border:1px solid #4b5365; border-radius:10px; padding:12px 14px; background:#181c25; color:#d8deeb; }
.toolbar { position:sticky; top:0; z-index:3; padding:12px clamp(16px,4vw,48px); background:rgba(16,17,20,.96); border-top:1px solid var(--line); border-bottom:1px solid var(--line); backdrop-filter:blur(8px); }
.toolbar-inner { max-width:1500px; margin:auto; display:grid; grid-template-columns:minmax(220px,2fr) repeat(5,minmax(125px,1fr)); gap:8px; }
input[type=search], select, button { font:inherit; border:1px solid var(--line); border-radius:8px; background:#20232b; color:#f2f4f8; padding:9px 10px; }
button { cursor:pointer; } button:hover { border-color:#65708a; }
.quick { max-width:1500px; margin:10px auto 0; display:flex; flex-wrap:wrap; gap:8px; }
main { max-width:1500px; margin:auto; padding:18px clamp(16px,4vw,48px) 360px; }
.group-title { margin:28px 0 10px; border-bottom:1px solid var(--line); padding-bottom:7px; }
.cards { display:grid; grid-template-columns:repeat(auto-fill,minmax(390px,1fr)); gap:12px; }
.card { display:grid; grid-template-columns:28px 142px 1fr; gap:10px; align-items:start; border:1px solid var(--line); border-radius:11px; padding:10px; background:var(--panel); }
.card.hidden { display:none; }
.card img { width:142px; max-height:110px; object-fit:contain; background:#fff; border-radius:6px; }
.thumb-fallback { width:142px; height:80px; display:flex; align-items:center; justify-content:center; color:#70798d; border:1px dashed #424958; border-radius:6px; font-size:12px; }
.task-title { font-weight:700; line-height:1.25; }
.task-id { font:12px ui-monospace,SFMono-Regular,Consolas,monospace; color:#c4ccda; overflow-wrap:anywhere; margin:3px 0 6px; }
.meta { display:flex; flex-wrap:wrap; gap:5px; margin-bottom:6px; }
.badge { font-size:11px; padding:2px 6px; border:1px solid #485166; border-radius:999px; color:#d6dceb; }
.star { border-color:#796b35; color:#ffe58f; }
.ply { border-color:#237f7d; color:#8de7df; background:#0b2929; }
.desc, .paper { font-size:12.5px; color:var(--muted); line-height:1.35; }
.paper { margin-top:6px; font-family:ui-monospace,SFMono-Regular,Consolas,monospace; }
.manual { opacity:.55; }
#selectionPanel { position:fixed; left:0; right:0; bottom:0; z-index:5; border-top:1px solid #475066; background:rgba(13,14,17,.98); padding:12px clamp(16px,4vw,48px); box-shadow:0 -12px 35px rgba(0,0,0,.35); }
.panel-inner { max-width:1500px; margin:auto; display:grid; grid-template-columns:1.1fr 1fr; gap:14px; }
.panel-title { display:flex; justify-content:space-between; gap:12px; align-items:center; }
.selected-numbers { color:var(--good); font-family:ui-monospace,SFMono-Regular,Consolas,monospace; margin:5px 0; }
.output { width:100%; min-height:86px; resize:vertical; font:12px ui-monospace,SFMono-Regular,Consolas,monospace; border:1px solid var(--line); border-radius:8px; background:#0b0c0f; color:#e7ebf4; padding:8px; }
.actions { display:flex; flex-wrap:wrap; gap:7px; margin-top:7px; }
.small { font-size:12px; color:var(--muted); }
@media (max-width:900px) { .toolbar-inner { grid-template-columns:1fr 1fr; } .panel-inner { grid-template-columns:1fr; } main { padding-bottom:520px; } }
@media (max-width:600px) { .cards { grid-template-columns:1fr; } .card { grid-template-columns:28px 1fr; } .card .preview { grid-column:2; } .toolbar-inner { grid-template-columns:1fr; } }
</style>
</head>
<body>
<header>
<h1>Interactive experiment configurator</h1>
<p>Choose the paper experiments you actually want. The numbers are visual shorthand; exported files use stable task IDs. Dependencies are resolved by <code>reproduce.py</code> at run time.</p>
<div class="warning"><strong>Safe by design:</strong> this page never edits <code>reproduction_config.json</code>. The downloaded selection contains only task IDs, so your output root, datasets, models, checkpoints, and device configured by <code>python reproduce.py setup</code> remain untouched. Save or move the downloaded JSON wherever convenient and adjust the path in the generated command if needed.</div>
</header>
<div class="toolbar"><div class="toolbar-inner">
<input id="search" type="search" placeholder="Search task, description, paper label…">
<select id="family"><option value="">All families</option></select>
<select id="etype"><option value="">M + I + mixed</option><option value="M">[M] measurement</option><option value="I">[I] intervention</option><option value="M/I">[M/I] mixed</option></select>
<select id="star"><option value="">All intervention families</option><option value="favorites">★ Show steering favorites only</option><option value="mu2">★ μ2 steering</option><option value="gpic">★ GPIC manifold surfing</option><option value="rn_surface">★ RN control surface</option><option value="none">No ★ family</option></select>
<select id="paper"><option value="">All paper sections</option></select>
<select id="model"><option value="">All model variants</option></select>
</div><div class="quick">
<button id="selectVisible">Select visible</button><button id="selectPaper">Select paper default</button><button id="selectAll">Select all automatic</button><button id="clear">Clear selection</button><label class="small"><input id="selectedOnly" type="checkbox"> show selected only</label>
</div></div>
<main id="taskRoot"></main>
<div id="selectionPanel"><div class="panel-inner">
<div>
<div class="panel-title"><strong id="selectionCount">0 selected</strong><span class="small">Generated from scratch from the current checkboxes — never merged with an older selection.</span></div>
<div class="selected-numbers" id="selectedNumbers">numbers: —</div>
<textarea class="output" id="selectionJson" readonly></textarea>
<div class="actions"><button id="copyJson">Copy JSON</button><button id="downloadJson">Download reproduction_selection.json</button><button id="copyIds">Copy task IDs</button><button id="copyNumbers">Copy UI numbers</button></div>
</div>
<div>
<div class="small">Normal run</div><textarea class="output" id="runCommand" readonly></textarea>
<div class="actions"><button id="copyRun">Copy run command</button></div>
<div class="small" style="margin-top:7px">Smoke first</div><textarea class="output" id="smokeCommand" readonly></textarea>
</div>
</div></div>
<script id="taskData" type="application/json">__TASK_DATA__</script>
<script>
const tasks = JSON.parse(document.getElementById('taskData').textContent);
const selected = new Set();
const root = document.getElementById('taskRoot');
const family = document.getElementById('family');
for (const f of [...new Set(tasks.map(t=>t.family))]) { const o=document.createElement('option'); o.value=f; o.textContent=f; family.appendChild(o); }
const paperSelect=document.getElementById('paper');
for (const s of [...new Set(tasks.flatMap(t=>t.paper_sections))].sort()) { const o=document.createElement('option'); o.value=s; o.textContent=s; paperSelect.appendChild(o); }
const modelSelect=document.getElementById('model');
for (const m of [...new Set(tasks.flatMap(t=>t.models))].sort()) { const o=document.createElement('option'); o.value=m; o.textContent=m; modelSelect.appendChild(o); }
const familyTitles = {bridge:'Bridge / trained-model behavior',workspace:'Native workspace / register circuit',rn:'RN mechanism / universality',figures:'Figure/postprocess tasks',conv1:'Conv1 / early routing',audit:'Utility / audit'};
function makeCard(t) {
  const card=document.createElement('div'); card.className='card'+(t.selectable?'':' manual'); card.dataset.id=t.id;
  const check=document.createElement('input'); check.type='checkbox'; check.disabled=!t.selectable; check.addEventListener('change',()=>{check.checked?selected.add(t.id):selected.delete(t.id); update();});
  const preview=document.createElement('div'); preview.className='preview';
  if (t.thumbnail) { const img=document.createElement('img'); img.src='figures_thumbs/'+t.thumbnail; img.alt=t.title; const fb=document.createElement('div'); fb.className='thumb-fallback'; fb.textContent='thumbnail not generated'; fb.style.display='none'; img.onerror=()=>{img.style.display='none'; fb.style.display='flex';}; preview.append(img,fb); }
  else { const fb=document.createElement('div'); fb.className='thumb-fallback'; fb.textContent='no figure'; preview.append(fb); }
  const body=document.createElement('div');
  const title=document.createElement('div'); title.className='task-title'; title.textContent=`#${t.number}  ${t.title}`;
  const id=document.createElement('div'); id.className='task-id'; id.textContent=t.id;
  const meta=document.createElement('div'); meta.className='meta';
  for (const x of [t.type_label,t.star_label,t.tier,t.kind].filter(Boolean)) { const b=document.createElement('span'); b.className='badge'+(String(x).startsWith('★')?' star':''); b.textContent=x; meta.append(b); }
  for (const tag of (t.special_tags || [])) { const b=document.createElement('span'); b.className='badge '+(String(tag).toUpperCase()==='PLY'?'ply':''); b.textContent=`(${tag})`; meta.append(b); }
  const desc=document.createElement('div'); desc.className='desc'; desc.textContent=t.description;
  const paper=document.createElement('div'); paper.className='paper'; paper.textContent=t.paper_sections.join(' · ') || 'support / utility';
  const models=document.createElement('div'); models.className='paper'; models.textContent=t.models.length ? ('models: '+t.models.join(' · ')) : '';
  body.append(title,id,meta,desc,paper,models); card.append(check,preview,body); card._check=check; return card;
}
const byFamily=new Map();
for (const t of tasks) { if(!byFamily.has(t.family)) byFamily.set(t.family,[]); byFamily.get(t.family).push(t); }
for (const [fam, rows] of byFamily) { const sec=document.createElement('section'); sec.dataset.family=fam; const h=document.createElement('h2'); h.className='group-title'; h.textContent=familyTitles[fam]||fam; const cards=document.createElement('div'); cards.className='cards'; for(const t of rows) cards.append(makeCard(t)); sec.append(h,cards); root.append(sec); }
function matches(t) {
  const q=document.getElementById('search').value.trim().toLowerCase();
  const f=family.value, et=document.getElementById('etype').value, st=document.getElementById('star').value, ps=document.getElementById('paper').value, mv=document.getElementById('model').value, only=document.getElementById('selectedOnly').checked;
  if(f && t.family!==f) return false; if(et && t.experiment_type!==et) return false;
  if(st==='none' && t.intervention_family) return false;
  if(st==='favorites' && !t.intervention_family) return false;
  if(st && !['none','favorites'].includes(st) && t.intervention_family!==st) return false;
  if(ps && !t.paper_sections.includes(ps)) return false; if(mv && !t.models.includes(mv)) return false; if(only && !selected.has(t.id)) return false;
  if(q) { const hay=[t.id,t.title,t.description,...t.paper_sections,...t.paper_artifacts,...t.models,...(t.special_tags||[])].join(' ').toLowerCase(); if(!hay.includes(q)) return false; }
  return true;
}
function applyFilter(){ for(const t of tasks){ const c=document.querySelector(`.card[data-id="${CSS.escape(t.id)}"]`); c.classList.toggle('hidden',!matches(t)); } for(const sec of root.querySelectorAll('section')) sec.style.display=sec.querySelector('.card:not(.hidden)')?'':'none'; }
function current(){ return tasks.filter(t=>selected.has(t.id)); }
function update(){
  applyFilter(); const rows=current(); document.getElementById('selectionCount').textContent=`${rows.length} selected`; document.getElementById('selectedNumbers').textContent=rows.length?('numbers: '+rows.map(t=>t.number).join(',')):'numbers: —';
  const obj={schema:1,name:'ModeMUX paper experiment selection',tasks:rows.map(t=>t.id)}; document.getElementById('selectionJson').value=JSON.stringify(obj,null,2)+'\n';
  const file='reproduction_selection.json'; const normal=`python reproduce.py run --selection ${file}`; document.getElementById('runCommand').value=normal; document.getElementById('smokeCommand').value=normal+' --smoke';
}
function setSelected(pred){ selected.clear(); for(const t of tasks){ if(t.selectable && pred(t)) selected.add(t.id); const c=document.querySelector(`.card[data-id="${CSS.escape(t.id)}"]`); if(c) c._check.checked=selected.has(t.id); } update(); }
for(const id of ['search','family','etype','star','paper','model','selectedOnly']) document.getElementById(id).addEventListener(id==='search'?'input':'change',applyFilter);
document.getElementById('selectVisible').onclick=()=>{for(const t of tasks){if(t.selectable&&matches(t)){selected.add(t.id);document.querySelector(`.card[data-id="${CSS.escape(t.id)}"]`)._check.checked=true;}}update();};
document.getElementById('selectPaper').onclick=()=>setSelected(t=>['canonical','figure'].includes(t.tier));
document.getElementById('selectAll').onclick=()=>setSelected(()=>true); document.getElementById('clear').onclick=()=>setSelected(()=>false);
async function copyText(id){try{await navigator.clipboard.writeText(document.getElementById(id).value);}catch(e){const el=document.getElementById(id);el.focus();el.select();document.execCommand('copy');}}
document.getElementById('copyJson').onclick=()=>copyText('selectionJson'); document.getElementById('copyRun').onclick=()=>copyText('runCommand');
document.getElementById('copyIds').onclick=async()=>{const s=current().map(t=>t.id).join('\n');try{await navigator.clipboard.writeText(s);}catch(e){prompt('Copy task IDs:',s);}};
document.getElementById('copyNumbers').onclick=async()=>{const s=current().map(t=>t.number).join(',');try{await navigator.clipboard.writeText(s);}catch(e){prompt('Copy UI numbers:',s);}};
document.getElementById('downloadJson').onclick=()=>{const blob=new Blob([document.getElementById('selectionJson').value],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='reproduction_selection.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);};
update();
</script>
</body>
</html>'''
    return template.replace("__TASK_DATA__", payload)


def main() -> int:
    rows = catalog_rows()
    (HERE / "task_catalog.json").write_text(json.dumps({"schema": 1, "tasks": rows}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (HERE / "README.md").write_text(render_readme(rows), encoding="utf-8", newline="\n")
    (HERE / "configurator.html").write_text(render_html(rows), encoding="utf-8", newline="\n")
    print(f"[done] generated {len(rows)} catalog entries in {HERE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
