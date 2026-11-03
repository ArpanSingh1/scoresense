"""ScoreSense on the school computer: tests, answer sheets cropped by question, marking and results.

    python eval/app.py        (or double-click ScoreSense_App.bat), then open http://localhost:8000

A test has its question paper, marks scheme and MCQ key, and the answer sheets of its students.
Each sheet is read (booklet.py), its MCQs checked against the key (mcq.py), and the teacher marks
the rest; the test page shows progress, the rank list and question-wise performance.

Everything stays on this computer, outside the code folder and OneDrive:
    <data>/tests/<test>/test.json        the test, its marks scheme and key; paper_* is its question paper
    <data>/<sheet>/                      one student's sheet: uploads/, crops/, result.json, marks.json,
                                         meta.json (test, student)
<data> is C:\\ocr_eval\\app_jobs (Windows) or ~/scoresense_data, or SCORESENSE_DATA if set.
Only this computer can open the page (it listens on 127.0.0.1). Sheets are read one at a time,
in upload order, since the model has the one GPU.
"""
import csv
import io
import json
import os
import queue
import re
import threading
import time
import uuid
import webbrowser
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import Body, FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response

from booklet import process
from mcq import check as check_mcq
from qpaper import read_scheme

PORT = 8000
JOBS_DIR = Path(os.environ.get("SCORESENSE_DATA") or
                (r"C:\ocr_eval\app_jobs" if os.name == "nt" else Path.home() / "scoresense_data"))
TESTS_DIR = JOBS_DIR / "tests"
PAGES = Path(__file__).parent / "app_pages"
ALLOWED = {".pdf", ".png", ".jpg", ".jpeg"}
ID = re.compile(r"^[0-9a-f]{12}$")
CROP_NAME = re.compile(r"^p\d{2}_(\d{2}|page)\.jpg$")
OPTIONS = ("A", "B", "C", "D")
TEST_FIELDS = ("name", "subject", "class_name", "date")

app = FastAPI()
jobs: dict[str, dict] = {}            # sheet id -> {"status", "message", "result", "files", "started"}
paper_jobs: dict[str, dict] = {}      # test id -> reading of its question paper
work: "queue.Queue[tuple[str, str, list[Path]]]" = queue.Queue()   # (kind, id, files), one at a time
lock = threading.Lock()               # guards test.json read-modify-write


# ---------- storage ----------

def new_id() -> str:
    return uuid.uuid4().hex[:12]


def read_json(path: Path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def test_path(test_id: str) -> Path:
    if not ID.match(test_id) or not (TESTS_DIR / test_id / "test.json").is_file():
        raise HTTPException(404, "No such test.")
    return TESTS_DIR / test_id / "test.json"


def load_test(test_id: str) -> dict:
    return read_json(test_path(test_id), {})


def save_test(test: dict) -> None:
    write_json(TESTS_DIR / test["id"] / "test.json", test)


def sheet_meta(sheet_id: str) -> dict:
    return read_json(JOBS_DIR / sheet_id / "meta.json", {})


def parse_scheme(text: str) -> dict[str, float]:
    """ "1-20:1, 21-24:2, 34.1:1" -> {"1": 1, ..., "24": 2, "34.1": 1}, the same rules as the pages."""
    out: dict[str, float] = {}
    for part in re.split(r"[,;\n]+", text or ""):
        m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(?:-\s*(\d+))?\s*[:=]\s*(\d+(?:\.\d+)?)\s*", part)
        if not m:
            continue
        if m.group(2):
            for n in range(int(float(m.group(1))), int(m.group(2)) + 1):
                out[str(n)] = float(m.group(3))
        else:
            out[m.group(1)] = float(m.group(3))
    return out


def out_of(scheme: dict[str, float], number: str) -> Optional[float]:
    if number in scheme:
        return scheme[number]
    main = number.split(".")[0]
    return scheme.get(main) if main.isdigit() else None


async def save_uploads(files: list[UploadFile], folder: Path, prefix: str = "") -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    saved = []
    for i, f in enumerate(files):
        if Path(f.filename or "").suffix.lower() not in ALLOWED:
            continue
        path = folder / f"{prefix}{i:02d}_{re.sub(r'[^A-Za-z0-9._ -]', '_', Path(f.filename).name)}"
        path.write_bytes(await f.read())
        saved.append(path)
    return saved


# ---------- the work queue: sheets, MCQ checks and question papers, one at a time ----------

def mcq_step(sheet_id: str, job: dict) -> None:
    """Check the sheet's multiple-choice answers against its test's key, if one is set."""
    test_id = sheet_meta(sheet_id).get("test_id")
    test = read_json(TESTS_DIR / test_id / "test.json", {}) if test_id else {}
    if not test.get("key"):
        return
    check_mcq(job["result"], JOBS_DIR / sheet_id / "crops", test.get("mcq", []), test["key"],
              progress=lambda m: job.__setitem__("message", m))
    job["result"]["mcq_checked"] = True
    write_json(JOBS_DIR / sheet_id / "result.json", job["result"])


def do_sheet(sheet_id: str, files: list[Path]) -> None:
    job = jobs[sheet_id]
    job["result"] = process(files, JOBS_DIR / sheet_id,
                            progress=lambda m: job.__setitem__("message", m),
                            partial=lambda r: job.__setitem__("result", r))
    mcq_step(sheet_id, job)


def do_paper(test_id: str, files: list[Path]) -> None:
    pj = paper_jobs[test_id]
    found = read_scheme(files, progress=lambda m: pj.__setitem__("message", m))
    with lock:
        test = load_test(test_id)
        keep = {k: v for k, v in test.get("key", {}).items() if int(k) in found["mcq"]}
        test.update({k: found[k] for k in ("sections", "scheme", "mcq", "max_marks", "problems", "found_on_page")},
                    key=keep, question_paper=[f.name[9:] for f in files])
        save_test(test)


def worker() -> None:
    while True:
        kind, item, files = work.get()
        state = paper_jobs[item] if kind == "paper" else jobs[item]
        state.update(status="working", started=time.time())
        try:
            if kind == "sheet":
                do_sheet(item, files)
            elif kind == "mcq":
                mcq_step(item, state)
            else:
                do_paper(item, files)
            state.update(status="done", message="")
        except Exception as e:  # shown on the page; the queue goes on with the next sheet
            state.update(status="failed", message=f"{type(e).__name__}: {e}")


def queue_sheet(sheet_id: str, files: list[Path]) -> None:
    jobs[sheet_id] = {"status": "queued", "message": "Waiting for its turn", "result": None,
                      "files": [p.name[3:] for p in files], "started": time.time()}
    work.put(("sheet", sheet_id, files))


def resume_unfinished() -> None:
    """Queue again the sheets that were waiting or being read when ScoreSense was last closed."""
    for test_dir in sorted(TESTS_DIR.glob("*/test.json"), key=lambda p: p.stat().st_mtime):
        for sheet_id in read_json(test_dir, {}).get("sheets", []):
            folder = JOBS_DIR / sheet_id
            if not (folder / "result.json").is_file() and (folder / "uploads").is_dir():
                files = sorted((folder / "uploads").iterdir())
                if files:
                    queue_sheet(sheet_id, files)


# ---------- sheets ----------

def get_job(sheet_id: str) -> dict:
    if not ID.match(sheet_id):
        raise HTTPException(404)
    if sheet_id not in jobs:                    # finished in an earlier run of the server
        saved = JOBS_DIR / sheet_id / "result.json"
        if not saved.is_file():
            if not (JOBS_DIR / sheet_id).is_dir():
                raise HTTPException(404, "No such answer sheet.")
            return {"status": "failed", "result": None, "files": [], "started": time.time(),
                    "message": "this sheet was stopped before it finished. Please upload it again."}
        jobs[sheet_id] = {"status": "done", "message": "", "files": [], "started": saved.stat().st_mtime,
                          "result": read_json(saved, None)}
    return jobs[sheet_id]


def needs_look(q: dict) -> bool:
    return bool(q.get("flags")) or bool(q.get("mcq") and q["mcq"]["verdict"] != "right")


def sheet_summary(sheet_id: str, scheme: dict[str, float]) -> dict:
    job = get_job(sheet_id) if (JOBS_DIR / sheet_id).is_dir() else {"status": "failed", "message": "missing"}
    meta = sheet_meta(sheet_id)
    marks = read_json(JOBS_DIR / sheet_id / "marks.json", {}).get("marks", {})
    result = job.get("result") or {}
    questions = [(i, q) for i, q in enumerate(result.get("questions", [])) if q["number"] != "start"]
    got = sum(m["mark"] or 0 for m in marks.values() if m.get("mark") is not None)
    marked = {k for k, m in marks.items() if m.get("mark") is not None}
    to_look = sum(1 for i, q in questions if str(i) not in marked and needs_look(q))
    unmarked = sum(1 for i, _ in questions if str(i) not in marked)
    done = job["status"] == "done"
    status = job["status"] if not done else ("completed" if questions and not unmarked else "review")
    pages = re.search(r"Page (\d+) of (\d+)", job.get("message", ""))
    return {"id": sheet_id, "student": meta.get("student") or sheet_id, "status": status,
            "message": job.get("message", ""), "files": meta.get("files", []),
            "progress": (int(pages.group(1)) / int(pages.group(2))) if pages else (1.0 if done else 0.0),
            "marks": got, "questions": len(questions), "unmarked": unmarked, "to_look": to_look,
            "uploaded": meta.get("uploaded", 0)}


@app.get("/job/{sheet_id}", response_class=HTMLResponse)
def job_page(sheet_id: str) -> str:
    get_job(sheet_id)
    return (PAGES / "results.html").read_text(encoding="utf-8")


@app.get("/api/job/{sheet_id}")
def job_status(sheet_id: str):
    job = get_job(sheet_id)
    meta = sheet_meta(sheet_id)
    test = read_json(TESTS_DIR / meta["test_id"] / "test.json", {}) if meta.get("test_id") else {}
    return JSONResponse({k: job[k] for k in ("status", "message", "result", "files")}
                        | {"seconds": round(time.time() - job["started"]), "student": meta.get("student", ""),
                           "test": {k: test.get(k) for k in ("id", "name", "scheme", "mcq", "key")}})


@app.get("/job/{sheet_id}/crops/{name}")
def crop(sheet_id: str, name: str):
    get_job(sheet_id)
    path = JOBS_DIR / sheet_id / "crops" / name
    if not CROP_NAME.match(name) or not path.is_file():
        raise HTTPException(404)
    return FileResponse(path)


@app.post("/api/job/{sheet_id}/check-mcq")
def check_mcq_again(sheet_id: str):
    """Check (again) a sheet read earlier, e.g. after the answer key was set or changed."""
    job = get_job(sheet_id)
    if job["status"] != "done" or not job.get("result"):
        raise HTTPException(409, "The sheet has not finished reading yet.")
    test_id = sheet_meta(sheet_id).get("test_id")
    if not test_id or not load_test(test_id).get("key"):
        raise HTTPException(400, "Set the answer key for this test first.")
    job.update(status="queued", message="Waiting for its turn")
    work.put(("mcq", sheet_id, []))
    return {"ok": True}


@app.get("/api/job/{sheet_id}/marks")
def get_marks(sheet_id: str):
    get_job(sheet_id)
    meta = sheet_meta(sheet_id)
    saved = read_json(JOBS_DIR / sheet_id / "marks.json", None)
    test = read_json(TESTS_DIR / meta["test_id"] / "test.json", {}) if meta.get("test_id") else {}
    if saved:
        saved["student"] = saved.get("student") or meta.get("student", "")
        return saved
    return {"scheme": test.get("scheme", ""), "student": meta.get("student", ""), "marks": {}}


@app.put("/api/job/{sheet_id}/marks")
def save_marks(sheet_id: str, data: dict = Body(...)):
    """{"scheme": "1-20:1, ...", "student": "...", "marks": {question index: {"number", "mark", "out_of", "by"}}}"""
    get_job(sheet_id)
    clean = {"scheme": str(data.get("scheme", ""))[:2000], "student": str(data.get("student", ""))[:200],
             "marks": {str(k): {"number": str(v.get("number", ""))[:20],
                                "mark": float(v["mark"]) if v.get("mark") is not None else None,
                                "out_of": float(v["out_of"]) if v.get("out_of") is not None else None,
                                "by": "model" if v.get("by") == "model" else "teacher"}
                       for k, v in dict(data.get("marks", {})).items()}}
    write_json(JOBS_DIR / sheet_id / "marks.json", clean)
    meta = sheet_meta(sheet_id)
    if meta and clean["student"] and clean["student"] != meta.get("student"):
        write_json(JOBS_DIR / sheet_id / "meta.json", {**meta, "student": clean["student"]})
    return {"ok": True}


def csv_response(rows: list[list], name: str) -> Response:
    out = io.StringIO()
    csv.writer(out).writerows(rows)
    # The byte-order mark makes Excel open it as UTF-8.
    return Response(chr(0xFEFF) + out.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{re.sub(r"[^A-Za-z0-9_-]", "_", name)}.csv"'})


@app.get("/job/{sheet_id}/marks.csv")
def marks_csv(sheet_id: str):
    get_job(sheet_id)
    data = read_json(JOBS_DIR / sheet_id / "marks.json", {"marks": {}, "student": ""})
    student = data["student"] or sheet_meta(sheet_id).get("student", "")
    rows = [["student", "question", "marks", "out_of", "decided_by"]]
    marks = sorted(data["marks"].items(), key=lambda kv: int(kv[0]))
    for _, m in marks:
        rows.append([student, m["number"], "" if m["mark"] is None else f"{m['mark']:g}",
                     "" if m["out_of"] is None else f"{m['out_of']:g}", "" if m["mark"] is None else m.get("by", "teacher")])
    rows.append([student, "TOTAL", f"{sum(m['mark'] or 0 for _, m in marks):g}",
                 f"{sum(m['out_of'] or 0 for _, m in marks):g}", ""])
    return csv_response(rows, f"marks_{student or sheet_id}")


# ---------- tests ----------

def test_overview(test: dict) -> dict:
    scheme = parse_scheme(test.get("scheme", ""))
    total = sum(scheme.values())
    sheets = [sheet_summary(s, scheme) for s in test.get("sheets", [])]
    finished = [s for s in sheets if s["status"] in ("completed", "review")]
    completed = [s for s in sheets if s["status"] == "completed"]
    pct = [100 * s["marks"] / total for s in completed] if total else []
    ranked = sorted(completed, key=lambda s: -s["marks"])
    for s in sheets:
        s["percent"] = round(100 * s["marks"] / total, 1) if total and s["status"] == "completed" else None
        s["rank"] = 1 + sum(o["marks"] > s["marks"] for o in ranked) if s["status"] == "completed" else None
    # Question-wise: marks given so far, per question number of the scheme (or as found on the sheets).
    perq: dict[str, dict] = {n: {"number": n, "out_of": o, "attempted": 0, "marked": 0, "sum": 0.0, "to_look": 0}
                             for n, o in scheme.items()}
    for s in finished:
        result = jobs.get(s["id"], {}).get("result") or {}
        marks = read_json(JOBS_DIR / s["id"] / "marks.json", {}).get("marks", {})
        for i, q in enumerate(result.get("questions", [])):
            if q["number"] in ("start", "?"):
                continue
            row = perq.setdefault(q["number"], {"number": q["number"], "out_of": out_of(scheme, q["number"]),
                                                "attempted": 0, "marked": 0, "sum": 0.0, "to_look": 0})
            row["attempted"] += 1
            m = marks.get(str(i), {})
            if m.get("mark") is not None:
                row["marked"] += 1
                row["sum"] += m["mark"]
            elif needs_look(q):
                row["to_look"] += 1
    questions = sorted(perq.values(), key=lambda r: [float(x) for x in r["number"].split(".") if x.isdigit()] or [1e9])
    for r in questions:
        r["average"] = round(r["sum"] / r["marked"], 2) if r["marked"] else None
        r["accuracy"] = round(100 * r["average"] / r["out_of"]) if r["average"] is not None and r["out_of"] else None
    return {**test, "total_marks": total, "sheets": sheets, "questions": questions,
            "paper_job": paper_jobs.get(test["id"], {"status": "idle", "message": ""}),
            "stats": {"uploaded": len(sheets), "read": len(finished), "completed": len(completed),
                      "average": round(sum(pct) / len(pct), 1) if pct else None,
                      "highest": round(max(pct), 1) if pct else None, "lowest": round(min(pct), 1) if pct else None}}


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return (PAGES / "home.html").read_text(encoding="utf-8")


@app.get("/api/tests")
def list_tests():
    out = []
    for path in TESTS_DIR.glob("*/test.json"):
        o = test_overview(read_json(path, {}))
        out.append({k: o.get(k) for k in ("id", "name", "subject", "class_name", "date", "created", "total_marks", "stats")})
    return sorted(out, key=lambda t: -(t.get("created") or 0))


@app.post("/api/tests")
def create_test(data: dict = Body(...)):
    name = str(data.get("name", "")).strip()[:200]
    if not name:
        raise HTTPException(400, "Give the test a name.")
    test = {"id": new_id(), "created": time.time(), "scheme": "", "mcq": [], "key": {}, "sheets": [],
            **{k: str(data.get(k, "")).strip()[:200] for k in TEST_FIELDS}, "name": name}
    save_test(test)
    return {"id": test["id"]}


@app.get("/test/{test_id}", response_class=HTMLResponse)
def test_page(test_id: str) -> str:
    test_path(test_id)
    return (PAGES / "test.html").read_text(encoding="utf-8")


@app.get("/test/{test_id}/setup", response_class=HTMLResponse)
def setup_page(test_id: str) -> str:
    test_path(test_id)
    return (PAGES / "setup.html").read_text(encoding="utf-8")


@app.get("/api/test/{test_id}")
def get_test(test_id: str):
    return test_overview(load_test(test_id))


@app.put("/api/test/{test_id}")
def edit_test(test_id: str, data: dict = Body(...)):
    """The teacher's edits: test details, marks scheme and MCQ key."""
    with lock:
        test = load_test(test_id)
        for k in TEST_FIELDS:
            if k in data:
                test[k] = str(data[k]).strip()[:200] or test.get(k, "")
        if "scheme" in data:
            test["scheme"] = str(data["scheme"])[:2000]
        if "mcq" in data:
            test["mcq"] = sorted({int(n) for n in data["mcq"] if 0 < int(n) < 1000})
        if "key" in data:
            test["key"] = {str(int(k)): v for k, v in dict(data["key"]).items() if v in OPTIONS}
        save_test(test)
    return {"ok": True}


@app.post("/test/{test_id}/paper")
async def upload_paper(test_id: str, files: list[UploadFile]):
    test_path(test_id)
    if paper_jobs.get(test_id, {}).get("status") in ("queued", "working"):
        raise HTTPException(409, "The question paper is already being read.")
    folder = TESTS_DIR / test_id
    for old in folder.glob("paper_*"):
        old.unlink()
    saved = await save_uploads(files, folder, prefix="paper_")
    if not saved:
        raise HTTPException(400, "Upload a PDF, PNG or JPG file.")
    paper_jobs[test_id] = {"status": "queued", "message": "Waiting for its turn", "started": time.time()}
    work.put(("paper", test_id, saved))
    return RedirectResponse(f"/test/{test_id}/setup", status_code=303)


@app.post("/test/{test_id}/sheets")
async def upload_sheets(test_id: str, files: list[UploadFile], mode: str = Form("each")):
    """mode "each": every file is one student's sheet, named after the file.
    mode "one": all the files are the pages of one student's sheet, in order."""
    test_path(test_id)
    groups = [[f] for f in files] if mode == "each" else [files]
    created = []
    for group in groups:
        sheet_id = new_id()
        saved = await save_uploads(group, JOBS_DIR / sheet_id / "uploads")
        if not saved:
            continue
        student = re.sub(r"[_]+", " ", Path(group[0].filename or "").stem).strip() if mode == "each" else ""
        write_json(JOBS_DIR / sheet_id / "meta.json", {"test_id": test_id, "student": student,
                                                       "files": [p.name[3:] for p in saved], "uploaded": time.time()})
        created.append((sheet_id, saved))
    if not created:
        raise HTTPException(400, "Upload PDF, PNG or JPG files.")
    with lock:
        test = load_test(test_id)
        test["sheets"] = test.get("sheets", []) + [s for s, _ in created]
        save_test(test)
    for sheet_id, saved in created:
        queue_sheet(sheet_id, saved)
    return RedirectResponse(f"/test/{test_id}", status_code=303)


@app.get("/test/{test_id}/results.csv")
def results_csv(test_id: str):
    o = test_overview(load_test(test_id))
    numbers = [q["number"] for q in o["questions"]]
    rows = [["rank", "student", *[f"Q{n}" for n in numbers], "total", "out_of", "percent", "status"]]
    for s in sorted(o["sheets"], key=lambda s: (s["rank"] is None, s["rank"] or 0, s["student"].lower())):
        result = jobs.get(s["id"], {}).get("result") or {}
        marks = read_json(JOBS_DIR / s["id"] / "marks.json", {}).get("marks", {})
        by_number: dict[str, float] = {}
        for i, q in enumerate(result.get("questions", [])):
            m = marks.get(str(i), {})
            if m.get("mark") is not None:
                by_number[q["number"]] = by_number.get(q["number"], 0) + m["mark"]
        rows.append([s["rank"] or "", s["student"], *[f"{by_number[n]:g}" if n in by_number else "" for n in numbers],
                     f"{s['marks']:g}", f"{o['total_marks']:g}", "" if s["percent"] is None else s["percent"],
                     {"completed": "marked", "review": "needs marking"}.get(s["status"], s["status"])])
    return csv_response(rows, f"results_{o.get('name') or test_id}")


@app.on_event("startup")
def start_worker() -> None:
    TESTS_DIR.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=worker, daemon=True).start()
    resume_unfinished()


if __name__ == "__main__":
    threading.Timer(1.5, lambda: webbrowser.open(f"http://localhost:{PORT}")).start()
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
