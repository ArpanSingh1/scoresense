"""ScoreSense on the school computer: upload an answer booklet, see its answers cropped by question.

    python eval/app.py        (or double-click ScoreSense_App.bat), then open http://localhost:8000

Uploads, crops and the teacher's marks (marks.json) stay on this computer, in
C:\\ocr_eval\\app_jobs\\<job>, outside OneDrive.
Only this computer can open the page (it listens on 127.0.0.1). One booklet is read at a time,
since the model has the one GPU; later uploads wait their turn.
"""
import csv
import io
import json
import os
import re
import threading
import time
import uuid
import webbrowser
from pathlib import Path

import uvicorn
from fastapi import Body, FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response

from booklet import process
from qpaper import read_scheme

PORT = 8000
# Student papers are kept here, outside the code folder (and outside OneDrive or any synced folder).
JOBS_DIR = Path(os.environ.get("SCORESENSE_DATA") or
                (r"C:\ocr_eval\app_jobs" if os.name == "nt" else Path.home() / "scoresense_data"))
PAGES = Path(__file__).parent / "app_pages"
ALLOWED = {".pdf", ".png", ".jpg", ".jpeg"}
JOB_ID = re.compile(r"^[0-9a-f]{12}$")
CROP_NAME = re.compile(r"^p\d{2}_(\d{2}|page)\.jpg$")
LAST_SCHEME = JOBS_DIR / "last_scheme.txt"        # the marks scheme last used, offered for the next paper
EXAM_DIR = JOBS_DIR / "exam"                       # the current exam: its question paper and exam.json
EXAM = EXAM_DIR / "exam.json"                      # {"name", "scheme", "mcq": [1, ...], "key": {"1": "B"}, ...}
OPTIONS = ("A", "B", "C", "D")

app = FastAPI()
jobs: dict[str, dict] = {}
exam_job = {"status": "idle", "message": ""}       # reading of the question paper, if one is going on
gpu = threading.Lock()


async def save_uploads(files: list[UploadFile], folder: Path) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    saved = []
    for i, f in enumerate(files):
        if Path(f.filename or "").suffix.lower() not in ALLOWED:
            continue
        path = folder / f"{i:02d}_{re.sub(r'[^A-Za-z0-9._ -]', '_', Path(f.filename).name)}"
        path.write_bytes(await f.read())
        saved.append(path)
    if not saved:
        raise HTTPException(400, "Upload PDF, PNG or JPG files.")
    return saved


def load_exam() -> dict:
    return json.loads(EXAM.read_text(encoding="utf-8")) if EXAM.is_file() else {}


def run_job(job_id: str, files: list[Path]) -> None:
    job = jobs[job_id]
    job["message"] = "Waiting for the booklet before this one to finish"
    with gpu:
        job["status"] = "working"
        try:
            job["result"] = process(files, JOBS_DIR / job_id,
                                    progress=lambda m: job.__setitem__("message", m),
                                    partial=lambda r: job.__setitem__("result", r))
            job["status"] = "done"
        except Exception as e:  # shown on the page; the server keeps running for the next upload
            job["status"], job["message"] = "failed", f"{type(e).__name__}: {e}"


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (PAGES / "upload.html").read_text(encoding="utf-8")


@app.post("/upload")
async def upload(files: list[UploadFile]):
    job_id = uuid.uuid4().hex[:12]
    saved = await save_uploads(files, JOBS_DIR / job_id / "uploads")
    jobs[job_id] = {"status": "queued", "message": "Starting", "result": None,
                    "files": [p.name[3:] for p in saved], "started": time.time()}
    threading.Thread(target=run_job, args=(job_id, saved), daemon=True).start()
    return RedirectResponse(f"/job/{job_id}", status_code=303)


@app.get("/exam", response_class=HTMLResponse)
def exam_page() -> str:
    return (PAGES / "exam.html").read_text(encoding="utf-8")


def run_exam(files: list[Path], name: str) -> None:
    exam_job.update(status="queued", message="Waiting for the paper being read to finish")
    with gpu:
        exam_job["status"] = "working"
        try:
            found = read_scheme(files, progress=lambda m: exam_job.__setitem__("message", m))
            old = load_exam()
            EXAM.write_text(json.dumps({**found, "name": name, "question_paper": [f.name[9:] for f in files],
                                        "key": {k: v for k, v in old.get("key", {}).items()
                                                if int(k) in found["mcq"]} if found["mcq"] else {}},
                                       indent=1), encoding="utf-8")
            exam_job.update(status="done", message="")
        except Exception as e:
            exam_job.update(status="failed", message=f"{type(e).__name__}: {e}")


@app.post("/exam/upload")
async def exam_upload(files: list[UploadFile]):
    if exam_job["status"] in ("queued", "working"):
        raise HTTPException(409, "A question paper is already being read.")
    for old in EXAM_DIR.glob("paper_*"):
        old.unlink()
    saved = await save_uploads(files, EXAM_DIR)
    saved = [p.rename(p.with_name("paper_" + p.name)) for p in saved]
    name = Path(saved[0].name[9:]).stem
    exam_job.update(status="queued", message="Starting")
    threading.Thread(target=run_exam, args=(saved, name), daemon=True).start()
    return RedirectResponse("/exam", status_code=303)


@app.get("/api/exam")
def get_exam():
    return {"job": exam_job, "exam": load_exam()}


@app.put("/api/exam")
def save_exam(data: dict = Body(...)):
    """The teacher's edits: name, scheme and MCQ key."""
    exam = load_exam()
    exam["name"] = str(data.get("name", exam.get("name", "")))[:200]
    exam["scheme"] = str(data.get("scheme", exam.get("scheme", "")))[:2000]
    exam["mcq"] = sorted({int(n) for n in data.get("mcq", exam.get("mcq", [])) if 0 < int(n) < 1000})
    exam["key"] = {str(int(k)): v for k, v in dict(data.get("key", {})).items() if v in OPTIONS}
    EXAM_DIR.mkdir(parents=True, exist_ok=True)
    EXAM.write_text(json.dumps(exam, indent=1), encoding="utf-8")
    return {"ok": True}


def get_job(job_id: str) -> dict:
    if not JOB_ID.match(job_id):
        raise HTTPException(404)
    if job_id not in jobs:                      # finished in an earlier run of the server
        saved = JOBS_DIR / job_id / "result.json"
        if not saved.is_file():
            if not (JOBS_DIR / job_id).is_dir():
                raise HTTPException(404, "No such booklet.")
            return {"status": "failed", "result": None, "files": [], "started": time.time(),
                    "message": "this paper was stopped before it finished, because ScoreSense was "
                               "closed. Please upload it again."}
        jobs[job_id] = {"status": "done", "message": "", "files": [], "started": saved.stat().st_mtime,
                        "result": json.loads(saved.read_text(encoding="utf-8"))}
    return jobs[job_id]


@app.get("/job/{job_id}", response_class=HTMLResponse)
def job_page(job_id: str) -> str:
    get_job(job_id)
    return (PAGES / "results.html").read_text(encoding="utf-8")


@app.get("/api/job/{job_id}")
def job_status(job_id: str):
    job = get_job(job_id)
    return JSONResponse({k: job[k] for k in ("status", "message", "result", "files")}
                        | {"seconds": round(time.time() - job["started"])})


@app.get("/job/{job_id}/crops/{name}")
def crop(job_id: str, name: str):
    get_job(job_id)
    if not CROP_NAME.match(name):
        raise HTTPException(404)
    path = JOBS_DIR / job_id / "crops" / name
    if not path.is_file():
        raise HTTPException(404)
    return FileResponse(path)


@app.get("/api/job/{job_id}/marks")
def get_marks(job_id: str):
    get_job(job_id)
    saved = JOBS_DIR / job_id / "marks.json"
    if saved.is_file():
        return json.loads(saved.read_text(encoding="utf-8"))
    scheme = load_exam().get("scheme") or (LAST_SCHEME.read_text(encoding="utf-8") if LAST_SCHEME.is_file() else "")
    return {"scheme": scheme, "student": "", "marks": {}}


@app.put("/api/job/{job_id}/marks")
def save_marks(job_id: str, data: dict = Body(...)):
    """{"scheme": "1-20:1, ...", "student": "...", "marks": {question index: {"number", "mark", "out_of"}}}"""
    get_job(job_id)
    clean = {"scheme": str(data.get("scheme", ""))[:2000], "student": str(data.get("student", ""))[:200],
             "marks": {str(k): {"number": str(v.get("number", ""))[:20],
                                "mark": float(v["mark"]) if v.get("mark") is not None else None,
                                "out_of": float(v["out_of"]) if v.get("out_of") is not None else None}
                       for k, v in dict(data.get("marks", {})).items()}}
    (JOBS_DIR / job_id / "marks.json").write_text(json.dumps(clean, indent=1), encoding="utf-8")
    if clean["scheme"]:
        LAST_SCHEME.write_text(clean["scheme"], encoding="utf-8")
    return {"ok": True}


@app.get("/job/{job_id}/marks.csv")
def marks_csv(job_id: str):
    get_job(job_id)
    saved = JOBS_DIR / job_id / "marks.json"
    data = json.loads(saved.read_text(encoding="utf-8")) if saved.is_file() else {"marks": {}, "student": ""}
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["student", "question", "marks", "out_of"])
    rows = sorted(data["marks"].items(), key=lambda kv: int(kv[0]))
    for _, m in rows:
        w.writerow([data["student"], m["number"], "" if m["mark"] is None else f"{m['mark']:g}",
                    "" if m["out_of"] is None else f"{m['out_of']:g}"])
    got = sum(m["mark"] or 0 for _, m in rows)
    total = sum(m["out_of"] or 0 for _, m in rows)
    w.writerow([data["student"], "TOTAL", f"{got:g}", f"{total:g}"])
    name = re.sub(r"[^A-Za-z0-9_-]", "_", data["student"] or job_id)
    # The byte-order mark makes Excel open it as UTF-8.
    return Response(chr(0xFEFF) + out.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="marks_{name}.csv"'})


if __name__ == "__main__":
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    threading.Timer(1.5, lambda: webbrowser.open(f"http://localhost:{PORT}")).start()
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
